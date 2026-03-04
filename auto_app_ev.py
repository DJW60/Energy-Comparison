# app.py
# Electricity plan comparator (Energex) - NEM12 5-minute CSV
#
# Features included:
# - Parse NEM12 200/300 records (5-min) into tidy intervals
# - Compare unlimited retailer plans (Flat or TOU) + Controlled load + FiT (flat or tiered)
# - TOU breakdown (kWh, %, rate, $) + flat-rate equivalent for General usage (E1)
# - Invoice-style line items (Daily charge, TOU bands, Controlled load, FiT credit)
# - Forecasting (1 month = repeat last 4 weeks; 12 months = replay last year if available, else fallback)
# - Plan Library (add/edit/duplicate/delete plans) persisted to plans.json next to app.py
# - Optional reconciliation: enter invoice total and see variance
#
# Run:
#   pip install streamlit pandas
#   python -m streamlit run app.py

from __future__ import annotations

import csv
import datetime as dt
import json
import copy
import io
import math
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _infer_interval_minutes(ts: pd.Series) -> float:
    """Infer the interval length (minutes) from a timestamp series."""
    if ts is None or len(ts) < 2:
        return 5.0
    s = pd.to_datetime(ts).sort_values()
    diffs = s.diff().dropna()
    if diffs.empty:
        return 5.0
    mins = diffs.dt.total_seconds().median() / 60.0
    if not mins or pd.isna(mins) or mins <= 0:
        return 5.0
    # Round to common NEM12 intervals (5 or 30)
    if abs(mins - 5) < 0.5:
        return 5.0
    if abs(mins - 30) < 1.0:
        return 30.0
    return float(mins)


def _interval_import_df(df_int: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-interval total import kWh indexed by timestamp.

    Supports both:
      - Long format: columns include ['timestamp','register','kwh']
      - Wide format: timestamp + register columns (e.g. E1/E2/...)
    """
    if df_int is None or len(df_int) == 0:
        return pd.DataFrame({"timestamp": pd.to_datetime([]), "import_kwh": []})

    d = df_int.copy()

    # Long format (most of the app uses this)
    if {"timestamp", "register", "kwh"}.issubset(d.columns):
        reg = d["register"].astype(str)
        imp = d.loc[reg.str.fullmatch(r"E\d+"), ["timestamp", "kwh"]].copy()
        if imp.empty:
            return pd.DataFrame({"timestamp": pd.to_datetime([]), "import_kwh": []})
        imp["timestamp"] = pd.to_datetime(imp["timestamp"])
        out = (
            imp.groupby("timestamp", as_index=False)["kwh"]
            .sum()
            .rename(columns={"kwh": "import_kwh"})
            .sort_values("timestamp")
        )
        return out

    # Wide format
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        import_cols = [c for c in d.columns if isinstance(c, str) and re.fullmatch(r"E\d+", c)]
        if not import_cols:
            return pd.DataFrame({"timestamp": pd.to_datetime([]), "import_kwh": []})
        out = d[["timestamp"] + import_cols].copy()
        out["import_kwh"] = out[import_cols].sum(axis=1, numeric_only=True)
        out = out[["timestamp", "import_kwh"]].sort_values("timestamp")
        return out

    return pd.DataFrame({"timestamp": pd.to_datetime([]), "import_kwh": []})


def compute_max_demand_kw(df_int: pd.DataFrame, window_minutes: int = 30) -> float:
    """Compute the maximum rolling-average demand (kW) over a window.

    We convert from energy (kWh per interval) to demand (kW) by:
      demand_kW = (kWh_in_window) * (60 / window_minutes)
    """
    imp = _interval_import_df(df_int)
    if imp.empty:
        return 0.0

    interval_min = _infer_interval_minutes(imp["timestamp"])
    if interval_min <= 0:
        interval_min = 5.0

    win = max(int(round(window_minutes / interval_min)), 1)
    kwh = pd.to_numeric(imp["import_kwh"], errors="coerce").fillna(0.0)

    roll_kwh = kwh.rolling(win, min_periods=win).sum()
    max_kwh = float(roll_kwh.max()) if len(roll_kwh) else 0.0

    return max_kwh * (60.0 / float(window_minutes))


def compute_monthly_max_demand(
    df_intervals: pd.DataFrame,
    window_minutes: int = 30,
) -> pd.DataFrame:
    """Monthly maximum demand (kW) using a rolling average window.

    Returns columns:
      - month: month start timestamp (datetime64[ns])
      - max_demand_kw: float
    """
    if df_intervals.empty:
        return pd.DataFrame(columns=["month", "max_demand_kw"])

    df = _interval_import_df(df_intervals)
    if df.empty:
        return pd.DataFrame(columns=["month", "max_demand_kw"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    window = max(int(window_minutes // 5), 1)

    # total import per 5-minute interval (kWh), then convert to kW for the averaging window:
    # avg_kW_over_window = (sum_kWh_over_window) * (60 / window_minutes)
    roll_kwh = df["import_kwh"].rolling(window=window, min_periods=window).sum()
    df["demand_kw"] = roll_kwh * (60.0 / float(window_minutes))

    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    out = (
        df.dropna(subset=["demand_kw"])
        .groupby("month", as_index=False)["demand_kw"]
        .max()
        .rename(columns={"demand_kw": "max_demand_kw"})
    )
    return out

def compute_average_daily_profile(df_int: pd.DataFrame, import_cols=("E1","E2")) -> pd.DataFrame:
    """Average daily profile by time-of-day (kW), plus weekday/weekend."""
    if df_int is None or df_int.empty or "timestamp" not in df_int.columns:
        return pd.DataFrame(columns=["tod","kw","day_type"])
    d = df_int.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    interval_min = _infer_interval_minutes(d["timestamp"])
    total_kwh = pd.Series(0.0, index=d.index)
    for c in import_cols:
        if c in d.columns:
            total_kwh = total_kwh + d[c].fillna(0.0).clip(lower=0.0)
    d["kw"] = total_kwh * (60.0 / interval_min)
    d["tod"] = d["timestamp"].dt.strftime("%H:%M")
    d["day_type"] = d["timestamp"].dt.dayofweek.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    # Average per time of day for each day type
    prof = d.groupby(["day_type","tod"])["kw"].mean().reset_index().sort_values(["day_type","tod"])
    return prof


def plot_average_24hr_profile_import_export(
    df_intervals: pd.DataFrame,
    solar_profile: Optional[pd.DataFrame] = None,
) -> None:
    """Average 24-hour profile across ALL days (hourly totals averaged by clock-hour).

    X-axis: 1-24 (1 = 00:00-01:00, 24 = 23:00-24:00)
    """
    import streamlit as st

    if df_intervals is None or df_intervals.empty:
        st.info("No interval data to plot.")
        return

    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce").fillna(0.0)

    # Import (E registers)
    imp = df[df["register"].astype(str).str.fullmatch(r"E\d+")]
    imp = imp.groupby("timestamp", as_index=False)["kwh"].sum()
    imp.rename(columns={"kwh": "import_kwh"}, inplace=True)

    # Export (B registers)
    exp = df[df["register"].astype(str).str.fullmatch(r"B\d+")]
    exp = exp.groupby("timestamp", as_index=False)["kwh"].sum()
    exp.rename(columns={"kwh": "export_kwh"}, inplace=True)

    merged = pd.merge(imp, exp, on="timestamp", how="outer").fillna(0.0)
    merged = merged.sort_values("timestamp")

    # Hourly totals first
    hourly_totals = (
        merged.set_index("timestamp")[["import_kwh", "export_kwh"]]
        .resample("1H")
        .sum()
        .reset_index()
    )

    hourly_totals["hour"] = hourly_totals["timestamp"].dt.hour

    hourly = (
        hourly_totals.groupby("hour", as_index=False)[["import_kwh", "export_kwh"]]
        .mean()
        .sort_values("hour")
    )

    hourly = (
        pd.DataFrame({"hour": range(24)})
        .merge(hourly, on="hour", how="left")
        .fillna(0.0)
    )

    include_solar = (
        isinstance(solar_profile, pd.DataFrame)
        and not solar_profile.empty
        and {"timestamp", "pv_kwh"}.issubset(solar_profile.columns)
    )
    if include_solar:
        sp = solar_profile.copy()
        sp["timestamp"] = pd.to_datetime(sp["timestamp"], errors="coerce")
        sp["pv_kwh"] = pd.to_numeric(sp["pv_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
        sp = sp.dropna(subset=["timestamp"])

        solar_hourly_totals = (
            sp.set_index("timestamp")[["pv_kwh"]]
            .resample("1H")
            .sum()
            .reset_index()
        )
        solar_hourly_totals["hour"] = solar_hourly_totals["timestamp"].dt.hour
        solar_hourly = (
            solar_hourly_totals.groupby("hour", as_index=False)[["pv_kwh"]]
            .mean()
            .rename(columns={"pv_kwh": "solar_kwh"})
        )
        hourly = hourly.merge(solar_hourly, on="hour", how="left")
        hourly["solar_kwh"] = pd.to_numeric(hourly["solar_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

    hourly["hour_1_24"] = hourly["hour"] + 1

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(hourly["hour_1_24"], hourly["import_kwh"], marker="o")
        ax.plot(hourly["hour_1_24"], hourly["export_kwh"], marker="o")
        if include_solar:
            ax.plot(hourly["hour_1_24"], hourly["solar_kwh"], marker="o")
            ax.fill_between(hourly["hour_1_24"], hourly["solar_kwh"], alpha=0.15)

        ax.set_xticks(range(1, 25))
        ax.set_xlim(1, 24)
        ax.set_xlabel("Hour of day (1-24)")
        ax.set_ylabel("Average kWh per hour")
        if include_solar:
            ax.set_title("Average 24-hour profile (Import, Export, and Solar production)")
        else:
            ax.set_title("Average 24-hour profile (Grid import & Grid export)")
        ax.grid(True, alpha=0.3)
        if include_solar:
            ax.legend(["Grid import (Usage)", "Grid export (Solar feed)", "Solar production"], loc="upper left")
        else:
            ax.legend(["Grid import (Usage)", "Grid export (Solar feed)"], loc="upper left")

        st.pyplot(fig, clear_figure=True)

    except Exception:
        chart_cols = ["import_kwh", "export_kwh"] + (["solar_kwh"] if include_solar else [])
        chart_df = hourly.set_index("hour_1_24")[chart_cols]
        st.line_chart(chart_df, height=320)

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If a merge creates *_dup columns, keep non-null values and drop the dup."""
    dup_cols = [c for c in df.columns if str(c).endswith("_dup")]
    for dc in dup_cols:
        base = str(dc)[:-4]
        if base in df.columns:
            df[base] = df[base].combine_first(df[dc])
            df.drop(columns=[dc], inplace=True)
        else:
            df.rename(columns={dc: base}, inplace=True)
    return df

import streamlit as st


# --- Scenario presets for homeowner-friendly sensitivity ---
SCENARIO_PRESETS = {
    "Conservative": {"price_growth": 0.03, "discount_rate": 0.05},
    "Typical": {"price_growth": 0.05, "discount_rate": 0.04},
    "High price growth": {"price_growth": 0.07, "discount_rate": 0.03},
    "Custom": None,
}

def apply_scenario_overrides(price_growth_rate: float, discount_rate: float):
    """Return (price_growth_rate, discount_rate, label) after applying the selected scenario preset."""
    preset = st.session_state.get("scenario_preset", "Typical")
    if preset in SCENARIO_PRESETS and SCENARIO_PRESETS[preset]:
        o = SCENARIO_PRESETS[preset]
        return float(o["price_growth"]), float(o["discount_rate"]), preset
    return float(price_growth_rate or 0.0), float(discount_rate or 0.0), preset


import datetime as dt
import re as _re

def _norm_days(s: str) -> str:
    return (str(s).strip().lower() if s is not None else "all")

def _norm_time_str(s: str) -> str:
    s = str(s).strip()
    # Accept '8:59' -> '08:59'
    if _re.match(r"^\d:\d\d$", s):
        s = "0" + s
    return s


# ---------------------------
# Plan library persistence (plans.json next to this file)
# ---------------------------
BASE_DIR = Path(__file__).parent
PLANS_FILE = BASE_DIR / "plans.json"
BATTERY_ASSUMPTIONS_FILE = BASE_DIR / "battery_assumptions.json"
LAST_PLANS_LOAD_STATUS = {
    "source": "defaults",
    "reason": "not_loaded",
    "count": 0,
    "skipped": 0,
}
LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS = {
    "source": "defaults",
    "reason": "not_loaded",
    "count": 0,
    "skipped": 0,
}

DEFAULT_BATTERY_ASSUMPTIONS_CONFIG = {
    "default_preset": "General Li-ion (typical)",
    "presets": [
        {
            "name": "General Li-ion (typical)",
            "cycle_life_efc": 6000.0,
            "eol_capacity_pct": 80.0,
            "source": "Internal default",
            "notes": "Generic baseline for modern residential Li-ion batteries.",
        },
        {
            "name": "Conservative aging",
            "cycle_life_efc": 5000.0,
            "eol_capacity_pct": 75.0,
            "source": "Internal default",
            "notes": "Stress-test assumptions when warranty evidence is limited.",
        },
        {
            "name": "High-cycle Li-ion",
            "cycle_life_efc": 8000.0,
            "eol_capacity_pct": 80.0,
            "source": "Internal default",
            "notes": "Optimistic assumption for strong cycling warranties.",
        },
    ],
}


# ---------------------------
# NEM12 parsing (5-min)
# ---------------------------
def read_nem12_5min(uploaded_file) -> pd.DataFrame:
    """
    Reads NEM12 CSV with 200/300 records.
    Returns tidy dataframe: register, timestamp, kwh
    Assumes 5-minute intervals => 288 values on each 300 line (common in NEM12).
    """
    if uploaded_file is None:
        return pd.DataFrame(columns=["register", "timestamp", "kwh"])

    rows = []
    current_register = None

    content = uploaded_file.getvalue().decode("utf-8", errors="replace").splitlines()
    reader = csv.reader(content)

    for rec in reader:
        if not rec:
            continue

        rec_type = rec[0].strip()

        if rec_type == "200":
            current_register = rec[3].strip() if len(rec) > 3 else None

        elif rec_type == "300" and current_register:
            if len(rec) < 3:
                continue
            try:
                day = dt.datetime.strptime(rec[1], "%Y%m%d").date()
            except Exception:
                continue

            vals = rec[2 : 2 + 288]
            base = dt.datetime.combine(day, dt.time(0, 0))

            for i, v in enumerate(vals):
                if v is None or v == "":
                    continue
                try:
                    kwh = float(v)
                except ValueError:
                    kwh = 0.0
                ts = base + dt.timedelta(minutes=5 * i)
                rows.append((current_register, ts, kwh))

    df = pd.DataFrame(rows, columns=["register", "timestamp", "kwh"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _read_tabular_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV/XLSX uploads into a DataFrame."""
    name = str(getattr(uploaded_file, "name", "") or "").lower()
    data = uploaded_file.getvalue()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))

    txt = data.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(txt))


def read_solar_profile_5min(uploaded_file) -> pd.DataFrame:
    """Parse solar production intervals into columns: timestamp, pv_kwh.

    Supports Fronius Solar.Web style exports such as:
      - Date and time
      - PV production
      - Optional unit row values like [dd.MM.yyyy HH:mm], [Wh]
    """
    raw = _read_tabular_upload(uploaded_file)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "pv_kwh"])

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {c: str(c).strip().lower() for c in df.columns}

    # Detect datetime column
    dt_col = None
    for c, lc in cols_lower.items():
        if "date" in lc and "time" in lc:
            dt_col = c
            break
    if dt_col is None:
        for c, lc in cols_lower.items():
            if "timestamp" in lc or "datetime" in lc:
                dt_col = c
                break
    if dt_col is None and len(df.columns) >= 1:
        dt_col = df.columns[0]

    # Detect PV column
    pv_col = None
    for c, lc in cols_lower.items():
        if ("pv" in lc and ("prod" in lc or "gen" in lc)) or ("solar" in lc and "prod" in lc):
            pv_col = c
            break
    if pv_col is None:
        for c, lc in cols_lower.items():
            if "production" in lc or "energy" in lc:
                pv_col = c
                break
    if pv_col is None:
        pv_candidates = [c for c in df.columns if c != dt_col]
        pv_col = pv_candidates[0] if pv_candidates else None

    if dt_col is None or pv_col is None:
        return pd.DataFrame(columns=["timestamp", "pv_kwh"])

    out = df[[dt_col, pv_col]].copy()
    out.columns = ["timestamp_raw", "pv_raw"]

    # Fronius exports use day-first like 01.12.2025 04:20
    ts = pd.to_datetime(out["timestamp_raw"], format="%d.%m.%Y %H:%M", errors="coerce")
    if ts.isna().any():
        ts2 = pd.to_datetime(out.loc[ts.isna(), "timestamp_raw"], dayfirst=True, errors="coerce")
        ts.loc[ts.isna()] = ts2
    out["timestamp"] = ts
    out["pv_num"] = pd.to_numeric(out["pv_raw"], errors="coerce")
    out = out.dropna(subset=["timestamp", "pv_num"])
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "pv_kwh"])

    pv_col_name = str(pv_col).lower()
    if "kwh" in pv_col_name:
        out["pv_kwh"] = out["pv_num"].astype(float)
    elif "wh" in pv_col_name:
        out["pv_kwh"] = out["pv_num"].astype(float) / 1000.0
    else:
        # Heuristic for unknown units: interval values > 20 are likely Wh, not kWh.
        q99 = float(out["pv_num"].quantile(0.99))
        out["pv_kwh"] = out["pv_num"].astype(float) / 1000.0 if q99 > 20.0 else out["pv_num"].astype(float)

    out["pv_kwh"] = out["pv_kwh"].clip(lower=0.0)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.groupby("timestamp", as_index=False)["pv_kwh"].sum().sort_values("timestamp")
    return out.reset_index(drop=True)


def align_solar_to_intervals(df_solar: pd.DataFrame, df_int: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Align solar production file data to model timestamps. Returns (aligned_df, match_pct)."""
    if (
        df_solar is None
        or df_solar.empty
        or df_int is None
        or df_int.empty
        or "timestamp" not in df_int.columns
        or "timestamp" not in df_solar.columns
        or "pv_kwh" not in df_solar.columns
    ):
        return pd.DataFrame(columns=["timestamp", "pv_kwh"]), 0.0

    base_ts = pd.DataFrame({"timestamp": pd.to_datetime(df_int["timestamp"], errors="coerce").dropna().drop_duplicates()})
    base_ts = base_ts.sort_values("timestamp").reset_index(drop=True)

    s = df_solar.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
    s["pv_kwh"] = pd.to_numeric(s["pv_kwh"], errors="coerce").fillna(0.0).astype(float)
    s = s.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False)["pv_kwh"].sum()

    m = base_ts.merge(s, on="timestamp", how="left")
    matched = float(m["pv_kwh"].notna().mean() * 100.0) if len(m) > 0 else 0.0
    m["pv_kwh"] = m["pv_kwh"].fillna(0.0).clip(lower=0.0)
    return m, matched


# ---------------------------
# Tariff models
# ---------------------------
@dataclass
class FlatTariff:
    cents_per_kwh: float


@dataclass
class TouBand:
    name: str
    cents_per_kwh: float
    days: str  # "all" | "wkday" | "wkend"
    start: str  # "HH:MM"
    end: str    # "HH:MM" (can cross midnight)


@dataclass
class TouTariff:
    bands: List[TouBand]


@dataclass
class TieredFiT:
    high_rate_cents: float      # e.g. 12
    high_kwh_per_day: float     # e.g. 14 kWh/day
    low_rate_cents: float       # e.g. 3


@dataclass
class Plan:
    name: str
    supply_cents_per_day: float
    import_type: str                  # "flat" or "tou"
    flat: Optional[FlatTariff] = None
    tou: Optional[TouTariff] = None

    # Controlled load
    controlled_supply_cents_per_day: float = 0.0
    controlled_cents_per_kwh: float = 0.0

    # Export / FiT
    feed_in_flat_cents_per_kwh: float = 0.0
    feed_in_tiered: Optional[TieredFiT] = None
    feed_in_tou: Optional[TouTariff] = None

    monthly_fee_cents: float = 0.0
    signup_credit_cents: float = 0.0


def _plan_to_dict(p: Plan) -> dict:
    d = {
        "name": p.name,
        "supply_cents_per_day": p.supply_cents_per_day,
        "import_type": p.import_type,
        "controlled_supply_cents_per_day": p.controlled_supply_cents_per_day,
        "controlled_cents_per_kwh": p.controlled_cents_per_kwh,
        "feed_in_flat_cents_per_kwh": p.feed_in_flat_cents_per_kwh,
        "monthly_fee_cents": p.monthly_fee_cents,
        "signup_credit_cents": p.signup_credit_cents,
        "flat": {"cents_per_kwh": p.flat.cents_per_kwh} if p.flat else None,
        "tou": {"bands": [b.__dict__ for b in p.tou.bands]} if p.tou else None,
        "feed_in_tiered": {
            "high_rate_cents": p.feed_in_tiered.high_rate_cents,
            "high_kwh_per_day": p.feed_in_tiered.high_kwh_per_day,
            "low_rate_cents": p.feed_in_tiered.low_rate_cents,
        } if p.feed_in_tiered else None,
        "feed_in_tou": {"bands": [b.__dict__ for b in p.feed_in_tou.bands]} if p.feed_in_tou else None,
    }
    return d


def _dict_to_plan(d: dict) -> Plan:
    flat = FlatTariff(**d["flat"]) if d.get("flat") else None
    tou_raw = d.get("tou") or {}
    tou = TouTariff(
        bands=[
            TouBand(
                name=str(b.get("name", "")).strip(),
                cents_per_kwh=float(b.get("cents_per_kwh", 0.0)),
                days=_norm_days(b.get("days", "all")),
                start=_norm_time_str(b.get("start", "00:00")),
                end=_norm_time_str(b.get("end", "00:00")),
            )
            for b in (tou_raw.get("bands") or [])
        ]
    ) if d.get("tou") else None
    fit = TieredFiT(**d["feed_in_tiered"]) if d.get("feed_in_tiered") else None
    fit_tou_raw = d.get("feed_in_tou") or {}
    fit_tou = TouTariff(
        bands=[
            TouBand(
                name=str(b.get("name", "")).strip(),
                cents_per_kwh=float(b.get("cents_per_kwh", 0.0)),
                days=_norm_days(b.get("days", "all")),
                start=_norm_time_str(b.get("start", "00:00")),
                end=_norm_time_str(b.get("end", "00:00")),
            )
            for b in (fit_tou_raw.get("bands") or [])
        ]
    ) if d.get("feed_in_tou") else None

    return Plan(
        name=d["name"],
        supply_cents_per_day=float(d.get("supply_cents_per_day", 0.0)),
        import_type=d.get("import_type", "flat"),
        flat=flat,
        tou=tou,
        controlled_supply_cents_per_day=float(d.get("controlled_supply_cents_per_day", 0.0)),
        controlled_cents_per_kwh=float(d.get("controlled_cents_per_kwh", 0.0)),
        feed_in_flat_cents_per_kwh=float(d.get("feed_in_flat_cents_per_kwh", 0.0)),
        feed_in_tiered=fit,
        feed_in_tou=fit_tou,
        monthly_fee_cents=float(d.get("monthly_fee_cents", 0.0)),
        signup_credit_cents=float(d.get("signup_credit_cents", 0.0)),
    )


def load_plans(defaults: list[Plan]) -> list[Plan]:
    """Load plans from plans.json.
    - If JSON is invalid, keep the broken file as a timestamped .broken copy and fall back to defaults.
    - If duplicate plan names exist, keep the first and rename later duplicates with ' (2)', ' (3)', ... so nothing disappears.
    """
    global LAST_PLANS_LOAD_STATUS

    if not PLANS_FILE.exists():
        LAST_PLANS_LOAD_STATUS = {
            "source": "defaults",
            "reason": "missing_file",
            "count": len(defaults),
            "skipped": 0,
        }
        return defaults

    raw = PLANS_FILE.read_text(encoding="utf-8")

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("plans.json must be a JSON array (list) of plans")
    except Exception:
        # Preserve the broken file so edits aren't lost
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        broken = PLANS_FILE.with_suffix(f".broken_{stamp}.json")
        try:
            broken.write_text(raw, encoding="utf-8")
        except Exception:
            pass
        LAST_PLANS_LOAD_STATUS = {
            "source": "defaults",
            "reason": "invalid_json",
            "count": len(defaults),
            "skipped": 0,
        }
        return defaults

    plans: list[Plan] = []
    name_counts: dict[str, int] = {}
    skipped = 0

    for item in data:
        try:
            p = _dict_to_plan(item)
        except Exception:
            skipped += 1
            continue

        base = (p.name or "").strip()
        if base == "":
            base = "Unnamed plan"

        n = name_counts.get(base, 0) + 1
        name_counts[base] = n

        if n == 1:
            p.name = base
        else:
            p.name = f"{base} ({n})"

        plans.append(p)

    if plans:
        LAST_PLANS_LOAD_STATUS = {
            "source": "file",
            "reason": "ok",
            "count": len(plans),
            "skipped": skipped,
        }
        return plans

    LAST_PLANS_LOAD_STATUS = {
        "source": "defaults",
        "reason": "no_valid_plans",
        "count": len(defaults),
        "skipped": skipped,
    }
    return defaults



def save_plans(plans: list[Plan]) -> None:
    data = [_plan_to_dict(p) for p in plans]
    PLANS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _default_battery_assumptions_config() -> dict:
    return copy.deepcopy(DEFAULT_BATTERY_ASSUMPTIONS_CONFIG)


def _normalize_battery_assumption_preset(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    try:
        cycle_life_efc = float(item.get("cycle_life_efc", 0.0))
        eol_capacity_pct = float(item.get("eol_capacity_pct", 0.0))
    except Exception:
        return None
    if cycle_life_efc <= 0:
        return None
    if eol_capacity_pct <= 0:
        return None
    return {
        "name": name,
        "cycle_life_efc": max(float(cycle_life_efc), 1.0),
        "eol_capacity_pct": max(min(float(eol_capacity_pct), 100.0), 1.0),
        "source": str(item.get("source", "")).strip(),
        "notes": str(item.get("notes", "")).strip(),
    }


def load_battery_assumptions_config() -> dict:
    """Load battery assumptions presets from battery_assumptions.json."""
    global LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS
    defaults = _default_battery_assumptions_config()

    if not BATTERY_ASSUMPTIONS_FILE.exists():
        LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS = {
            "source": "defaults",
            "reason": "missing_file",
            "count": len(defaults.get("presets", [])),
            "skipped": 0,
        }
        return defaults

    raw = BATTERY_ASSUMPTIONS_FILE.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("battery_assumptions.json must be a JSON object")
        presets_raw = data.get("presets")
        if not isinstance(presets_raw, list):
            raise ValueError("battery_assumptions.json.presets must be a list")
    except Exception:
        LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS = {
            "source": "defaults",
            "reason": "invalid_json",
            "count": len(defaults.get("presets", [])),
            "skipped": 0,
        }
        return defaults

    presets: list[dict] = []
    skipped = 0
    name_counts: dict[str, int] = {}
    for item in presets_raw:
        p = _normalize_battery_assumption_preset(item)
        if p is None:
            skipped += 1
            continue
        base = str(p["name"]).strip()
        n = name_counts.get(base, 0) + 1
        name_counts[base] = n
        if n > 1:
            p["name"] = f"{base} ({n})"
        presets.append(p)

    if not presets:
        LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS = {
            "source": "defaults",
            "reason": "no_valid_presets",
            "count": len(defaults.get("presets", [])),
            "skipped": skipped,
        }
        return defaults

    preset_names = [str(p.get("name", "")).strip() for p in presets]
    default_name = str(data.get("default_preset", "")).strip()
    if default_name not in preset_names:
        default_name = preset_names[0]

    LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS = {
        "source": "file",
        "reason": "ok",
        "count": len(presets),
        "skipped": skipped,
    }
    return {
        "default_preset": default_name,
        "presets": presets,
    }


# ---------------------------
# Register mapping (confirmed)
# ---------------------------
GENERAL_REGS = ["E1"]
CONTROLLED_REGS = ["E2"]
EXPORT_REGS = ["B1"]


# ---------------------------
# Default plans (GST inclusive)
# ---------------------------
ORIGIN = Plan(
    name="Origin",
    supply_cents_per_day=142.868,
    controlled_supply_cents_per_day=3.971,
    import_type="flat",
    flat=FlatTariff(cents_per_kwh=35.244),
    controlled_cents_per_kwh=17.765,
    # NOTE: default tier based on your earlier statement; adjust in Plan Library as needed
    feed_in_tiered=TieredFiT(high_rate_cents=10.0, high_kwh_per_day=10.0, low_rate_cents=3.0),
)

ALINTA = Plan(
    name="Alinta Energy (TOU)",
    supply_cents_per_day=117.007,
    import_type="tou",
    tou=TouTariff(bands=[
        # Weekdays
        TouBand("off peak",  24.706, "wkday", "00:00", "07:00"),
        TouBand("shoulder",  28.942, "wkday", "07:00", "16:00"),
        TouBand("peak",      38.455, "wkday", "16:00", "20:00"),
        TouBand("shoulder",  28.942, "wkday", "20:00", "22:00"),
        TouBand("off peak",  24.706, "wkday", "22:00", "00:00"),
        # Weekends
        TouBand("off peak",  24.706, "wkend", "22:00", "07:00"),
        TouBand("shoulder",  28.942, "wkend", "07:00", "22:00"),
    ]),
    controlled_cents_per_kwh=15.323,
    feed_in_flat_cents_per_kwh=4.0,
)


# ---------------------------
# TOU helpers
# ---------------------------
def _parse_hhmm(s: str):
    """Parse time strings robustly.
    Returns dt.time on success, or None if the value can't be parsed.
    Accepts 'HH:MM', 'H:MM', 'HH', 'H'. Blank/None -> None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None

    if ":" not in s:
        if s.isdigit():
            h = int(s)
            if 0 <= h <= 23:
                return dt.time(h, 0)
        return None

    parts = s.split(":")
    if len(parts) != 2:
        return None
    h_str, m_str = parts[0].strip(), parts[1].strip()
    if not h_str.isdigit():
        return None
    if m_str == "":
        m_str = "0"
    if not m_str.isdigit():
        return None
    h, m = int(h_str), int(m_str)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return dt.time(h, m)


    if ":" not in s:
        # allow '9' or '09'
        if s.isdigit():
            h = int(s)
            if not (0 <= h <= 23):
                raise ValueError(f"Hour out of range: {h}")
            return dt.time(h, 0)
        raise ValueError(f"Invalid time format: {s}")

    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {s}")

    h_str, m_str = parts[0].strip(), parts[1].strip()
    if h_str == "" or not h_str.isdigit():
        raise ValueError(f"Invalid hour: {s}")
    if m_str == "":
        m_str = "0"
    if not m_str.isdigit():
        raise ValueError(f"Invalid minute: {s}")

    h, m = int(h_str), int(m_str)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Time out of range: {s}")
    return dt.time(h, m)


def _is_weekday(ts: pd.Timestamp) -> bool:
    return ts.weekday() < 5


def _band_matches(ts: pd.Timestamp, band: TouBand) -> bool:
    t = ts.time()
    start = _parse_hhmm(band.start)
    end = _parse_hhmm(band.end)

    # If a TOU row has blank/invalid times, ignore it rather than crashing
    if start is None or end is None:
        return False

    if band.days == "wkday" and not _is_weekday(ts):
        return False
    if band.days == "wkend" and _is_weekday(ts):
        return False

    # Supports crossing midnight
    if start < end:
        return start <= t < end
    if start > end:
        return (t >= start) or (t < end)
    # start == end => full day
    return True


def tou_rate_for_ts(ts: pd.Timestamp, tou: TouTariff) -> float:
    for band in tou.bands:
        if _band_matches(ts, band):
            return float(band.cents_per_kwh)
    return 0.0


def tou_band_name_for_ts(ts: pd.Timestamp, tou: TouTariff) -> str:
    for band in tou.bands:
        if _band_matches(ts, band):
            return str(band.name)
    return "unrated"


def tou_breakdown_general(df_intervals: pd.DataFrame, tou: TouTariff) -> pd.DataFrame:
    """
    GENERAL usage (E1) split into TOU bands with kWh and cents.
    Returns: band, kwh, rate_c_per_kwh, cents
    """
    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["register"].isin(GENERAL_REGS)].copy()

    if df.empty:
        return pd.DataFrame(columns=["band", "kwh", "rate_c_per_kwh", "cents"])

    g = df.groupby("timestamp", as_index=False)["kwh"].sum()
    g["band"] = g["timestamp"].apply(lambda ts: tou_band_name_for_ts(pd.Timestamp(ts), tou))
    g["rate_c_per_kwh"] = g["timestamp"].apply(lambda ts: tou_rate_for_ts(pd.Timestamp(ts), tou))
    g["cents"] = g["kwh"] * g["rate_c_per_kwh"]

    out = g.groupby(["band", "rate_c_per_kwh"], as_index=False)[["kwh", "cents"]].sum()
    return out.sort_values("cents", ascending=False)


def _has_fit_tou(plan: Plan) -> bool:
    return bool(plan.feed_in_tou and plan.feed_in_tou.bands)


def fit_mode_for_plan(plan: Plan) -> str:
    if _has_fit_tou(plan):
        return "tou"
    if plan.feed_in_tiered:
        return "tiered"
    return "flat"


def fit_rate_for_ts(ts: pd.Timestamp, plan: Plan) -> float:
    """
    Export FiT rate at a timestamp.
    If FiT TOU bands exist, they override the base flat FiT for matching windows.
    """
    base = float(plan.feed_in_flat_cents_per_kwh or 0.0)
    if not _has_fit_tou(plan):
        return base
    for band in plan.feed_in_tou.bands:
        if _band_matches(ts, band):
            return float(band.cents_per_kwh)
    return base


def fit_band_name_for_ts(ts: pd.Timestamp, plan: Plan) -> str:
    if not _has_fit_tou(plan):
        return "flat"
    for band in plan.feed_in_tou.bands:
        if _band_matches(ts, band):
            return str(band.name)
    return "base"


def fit_breakdown_export(df_intervals: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    """
    EXPORT (B1) split into FiT bands with kWh and cents.
    For FiT TOU, non-matching windows fall back to the plan's base flat FiT rate.
    Returns: band, kwh, rate_c_per_kwh, cents
    """
    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["register"].isin(EXPORT_REGS)].copy()

    if df.empty:
        return pd.DataFrame(columns=["band", "kwh", "rate_c_per_kwh", "cents"])

    g = df.groupby("timestamp", as_index=False)["kwh"].sum()
    g["band"] = g["timestamp"].apply(lambda ts: fit_band_name_for_ts(pd.Timestamp(ts), plan))
    g["rate_c_per_kwh"] = g["timestamp"].apply(lambda ts: fit_rate_for_ts(pd.Timestamp(ts), plan))
    g["cents"] = g["kwh"] * g["rate_c_per_kwh"]

    out = g.groupby(["band", "rate_c_per_kwh"], as_index=False)[["kwh", "cents"]].sum()
    return out.sort_values("cents", ascending=False)


def _is_night_timestamp(ts: pd.Timestamp) -> bool:
    h = int(ts.hour)
    return (h < 6) or (h >= 18)


def _time_range_overlaps_night(start: dt.time, end: dt.time) -> bool:
    def _m(t: dt.time) -> int:
        return int(t.hour) * 60 + int(t.minute)

    night_ranges = [(0, 6 * 60), (18 * 60, 1440)]
    s = _m(start)
    e = _m(end)
    # start == end means full day
    ranges = [(0, 1440)] if s == e else ([(s, e)] if s < e else [(s, 1440), (0, e)])

    for rs, re in ranges:
        for ns, ne in night_ranges:
            if rs < ne and re > ns:
                return True
    return False


def _fit_tou_has_night_bonus_rows(rows: list[dict], base_rate_cents: float) -> bool:
    base = float(base_rate_cents or 0.0)
    for r in rows:
        try:
            band_rate = float(r.get("cents_per_kwh", 0.0) or 0.0)
        except Exception:
            band_rate = 0.0
        if band_rate <= base:
            continue
        start = _parse_hhmm(_norm_time_str(r.get("start", "")))
        end = _parse_hhmm(_norm_time_str(r.get("end", "")))
        if start is None or end is None:
            continue
        if _time_range_overlaps_night(start, end):
            return True
    return False


def fit_tou_night_bonus_export_kwh(df_intervals: pd.DataFrame, plan: Plan) -> float:
    """
    Export kWh that actually lands in bonus FiT night windows (rate > base, night hours).
    This is a practical proxy for 'battery-needed' FiT benefit realization.
    """
    if not _has_fit_tou(plan):
        return 0.0
    base = float(plan.feed_in_flat_cents_per_kwh or 0.0)
    rows = [b.__dict__ for b in (plan.feed_in_tou.bands if plan.feed_in_tou else [])]
    if not _fit_tou_has_night_bonus_rows(rows, base):
        return 0.0

    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["register"].isin(EXPORT_REGS)].copy()
    if df.empty:
        return 0.0

    g = df.groupby("timestamp", as_index=False)["kwh"].sum()
    g["rate"] = g["timestamp"].apply(lambda ts: fit_rate_for_ts(pd.Timestamp(ts), plan))
    g["is_night"] = g["timestamp"].apply(lambda ts: _is_night_timestamp(pd.Timestamp(ts)))
    g["is_bonus"] = g["rate"] > base
    return float(g.loc[g["is_night"] & g["is_bonus"], "kwh"].sum())


def _battery_dependent_plan_reason(plan: Plan) -> str | None:
    """Return a plain-language reason when a plan appears battery-targeted."""
    base = float(plan.feed_in_flat_cents_per_kwh or 0.0)
    fit_rows = [b.__dict__ for b in (plan.feed_in_tou.bands if plan.feed_in_tou else [])]
    if _fit_tou_has_night_bonus_rows(fit_rows, base):
        return "Night FiT bonus windows are typically only achievable with battery export."

    nm = str(plan.name or "").strip().lower()
    if any(tok in nm for tok in ("battery", "powerwall", "vpp", "virtual power")):
        return "Plan name indicates a battery-targeted offer."
    return None


def _flat_plan_uses_single_import_rate(plan: Plan) -> bool:
    """True when a flat plan has no separate controlled-load usage rate."""
    if plan.import_type != "flat":
        return False
    try:
        return float(plan.controlled_cents_per_kwh or 0.0) <= 0.0
    except Exception:
        return True


# ---------------------------
# Simulation
# ---------------------------
def simulate_plan(df_intervals: pd.DataFrame, plan: Plan, include_signup_credit: bool = False) -> dict:
    if df_intervals.empty:
        return {
            "days": 0,
            "general_kwh": 0.0,
            "controlled_kwh": 0.0,
            "export_kwh": 0.0,
            "supply_cents": 0.0,
            "controlled_supply_cents": 0.0,
            "monthly_fee_cents": 0.0,
            "general_cents": 0.0,
            "controlled_cents": 0.0,
            "import_kwh": 0.0,
            "import_cents": 0.0,
            "flat_applies_to_all_import": False,
            "export_credit_cents": 0.0,
            "signup_credit_cents_available": float(plan.signup_credit_cents or 0.0),
            "signup_credit_cents_applied": 0.0,
            "total_cents": 0.0,
            "line_items": pd.DataFrame(columns=["item", "qty", "unit", "rate", "amount"]),
        }

    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    reg = df["register"].astype(str)
    kwh = pd.to_numeric(df["kwh"], errors="coerce").fillna(0.0).astype(float)
    df["general_kwh"] = kwh.where(reg.isin(GENERAL_REGS), 0.0)
    df["controlled_kwh"] = kwh.where(reg.isin(CONTROLLED_REGS), 0.0)
    df["export_kwh"] = kwh.where(reg.isin(EXPORT_REGS), 0.0)

    g = df.groupby("timestamp", as_index=False)[["general_kwh", "controlled_kwh", "export_kwh"]].sum()
    g["date"] = g["timestamp"].dt.date
    n_days = int(g["date"].nunique())
    total_general_kwh = float(g["general_kwh"].sum())
    total_controlled_kwh = float(g["controlled_kwh"].sum())
    controlled_rate_c = float(plan.controlled_cents_per_kwh or 0.0)
    flat_applies_to_all_import = _flat_plan_uses_single_import_rate(plan)

    # General import cents
    if plan.import_type == "flat":
        rate_c = float(plan.flat.cents_per_kwh if plan.flat else 0.0)
        general_cents = total_general_kwh * rate_c
        controlled_cents = total_controlled_kwh * (rate_c if flat_applies_to_all_import else controlled_rate_c)
    else:
        rates = [tou_rate_for_ts(pd.Timestamp(ts), plan.tou) for ts in g["timestamp"]]
        g["rate_cents_per_kwh"] = rates
        general_cents = float((g["general_kwh"] * g["rate_cents_per_kwh"]).sum())
        controlled_cents = total_controlled_kwh * controlled_rate_c

    # Supply cents
    supply_cents = n_days * float(plan.supply_cents_per_day or 0.0)
    controlled_supply_cents = n_days * float(plan.controlled_supply_cents_per_day or 0.0)

    # Monthly fee (scaled)
    monthly_fee_cents = float(plan.monthly_fee_cents or 0.0) * (n_days / 30.4375)
    signup_credit_cents_available = float(plan.signup_credit_cents or 0.0)
    signup_credit_cents_applied = signup_credit_cents_available if include_signup_credit else 0.0

    # Export credits
    total_export_kwh = float(g["export_kwh"].sum())
    if _has_fit_tou(plan):
        export_detail = df[df["register"].isin(EXPORT_REGS)].groupby("timestamp", as_index=False)["kwh"].sum()
        if export_detail.empty:
            export_credit_cents = 0.0
        else:
            export_detail["rate_c_per_kwh"] = export_detail["timestamp"].apply(
                lambda ts: fit_rate_for_ts(pd.Timestamp(ts), plan)
            )
            export_credit_cents = float((export_detail["kwh"] * export_detail["rate_c_per_kwh"]).sum())
    elif plan.feed_in_tiered:
        cap_kwh = n_days * float(plan.feed_in_tiered.high_kwh_per_day)
        high_kwh = min(total_export_kwh, cap_kwh)
        low_kwh = max(total_export_kwh - cap_kwh, 0.0)
        export_credit_cents = (
            high_kwh * float(plan.feed_in_tiered.high_rate_cents)
            + low_kwh * float(plan.feed_in_tiered.low_rate_cents)
        )
    else:
        export_credit_cents = total_export_kwh * float(plan.feed_in_flat_cents_per_kwh or 0.0)

    total_cents = (
        supply_cents
        + controlled_supply_cents
        + monthly_fee_cents
        + general_cents
        + controlled_cents
        - export_credit_cents
        - signup_credit_cents_applied
    )

    # Invoice-style breakdown lines
    line_items = []

    line_items.append({
        "item": "Daily Charge",
        "qty": n_days,
        "unit": "days",
        "rate": float(plan.supply_cents_per_day or 0.0) / 100.0,
        "amount": supply_cents / 100.0,
    })

    if float(plan.controlled_supply_cents_per_day or 0.0) > 0:
        line_items.append({
            "item": "Controlled Supply",
            "qty": n_days,
            "unit": "days",
            "rate": float(plan.controlled_supply_cents_per_day) / 100.0,
            "amount": controlled_supply_cents / 100.0,
        })

    if plan.import_type == "flat":
        flat_rate = float(plan.flat.cents_per_kwh if plan.flat else 0.0)
        if flat_applies_to_all_import:
            line_items.append({
                "item": "Import Usage (single rate)",
                "qty": total_general_kwh + total_controlled_kwh,
                "unit": "kWh",
                "rate": flat_rate / 100.0,
                "amount": (general_cents + controlled_cents) / 100.0,
            })
        else:
            line_items.append({
                "item": "General Usage",
                "qty": total_general_kwh,
                "unit": "kWh",
                "rate": flat_rate / 100.0,
                "amount": general_cents / 100.0,
            })
    else:
        bd = tou_breakdown_general(df_intervals, plan.tou)
        for _, r in bd.iterrows():
            line_items.append({
                "item": str(r["band"]).title(),
                "qty": float(r["kwh"]),
                "unit": "kWh",
                "rate": float(r["rate_c_per_kwh"]) / 100.0,
                "amount": float(r["cents"]) / 100.0,
            })

    if (plan.import_type != "flat" or not flat_applies_to_all_import) and float(plan.controlled_cents_per_kwh or 0.0) > 0:
        line_items.append({
            "item": "Controlled Load",
            "qty": total_controlled_kwh,
            "unit": "kWh",
            "rate": controlled_rate_c / 100.0,
            "amount": controlled_cents / 100.0,
        })

    if total_export_kwh > 0:
        if _has_fit_tou(plan):
            fit_bd = fit_breakdown_export(df_intervals, plan)
            for _, r in fit_bd.iterrows():
                band_name = str(r["band"]).strip() or "base"
                line_items.append({
                    "item": f"Solar Export ({band_name})",
                    "qty": float(r["kwh"]),
                    "unit": "kWh",
                    "rate": -(float(r["rate_c_per_kwh"]) / 100.0),
                    "amount": -(float(r["cents"]) / 100.0),
                })
        else:
            eff_fit_rate = (export_credit_cents / total_export_kwh) / 100.0 if total_export_kwh else 0.0
            line_items.append({
                "item": "Solar Export (FiT)",
                "qty": total_export_kwh,
                "unit": "kWh",
                "rate": -eff_fit_rate,
                "amount": -(export_credit_cents / 100.0),
            })

    if signup_credit_cents_applied > 0:
        line_items.append({
            "item": "One-time Sign-up Credit",
            "qty": 1,
            "unit": "credit",
            "rate": -(signup_credit_cents_applied / 100.0),
            "amount": -(signup_credit_cents_applied / 100.0),
        })

    return {
        "days": n_days,
        "general_kwh": total_general_kwh,
        "controlled_kwh": total_controlled_kwh,
        "export_kwh": total_export_kwh,
        "supply_cents": supply_cents,
        "controlled_supply_cents": controlled_supply_cents,
        "monthly_fee_cents": monthly_fee_cents,
        "general_cents": general_cents,
        "controlled_cents": controlled_cents,
        "import_kwh": total_general_kwh + total_controlled_kwh,
        "import_cents": general_cents + controlled_cents,
        "flat_applies_to_all_import": flat_applies_to_all_import,
        "export_credit_cents": export_credit_cents,
        "signup_credit_cents_available": signup_credit_cents_available,
        "signup_credit_cents_applied": signup_credit_cents_applied,
        "total_cents": total_cents,
        "line_items": pd.DataFrame(line_items),
    }



# =========================
# Battery simulation (Level 1 dispatch, no grid charging)
# =========================

@dataclass
class BatteryParams:
    """Simple behind-the-meter battery model (Level 1).

    Level 1 assumptions:
    - Charging is ONLY from PV surplus (i.e., it reduces export). No grid charging.
    - Discharging offsets imports, prioritising general (E1) then controlled load (E2).
    - Optional discharge threshold (c/kWh): only discharge when current import rate >= threshold.
      If not provided and the plan is TOU, we default to the highest-rate band (or a band with
      'peak' in its name if present).
    """

    capacity_kwh: float
    power_kw: float = 5.0
    roundtrip_eff: float = 0.90
    reserve_frac: float = 0.10   # keep this fraction as reserve (not discharged)
    initial_soc_frac: float = 0.50
    discharge_min_rate_cents: float | None = None
    charge_from_export_only: bool = True  # keep grid charging OFF


def _split_roundtrip_eff(roundtrip_eff: float) -> tuple[float, float]:
    """Return (charge_eff, discharge_eff) assuming symmetric losses."""
    rt = float(roundtrip_eff or 1.0)
    rt = max(min(rt, 1.0), 0.01)
    e = rt ** 0.5
    return e, e




def _npv(cashflows: list[float], discount_rate: float) -> float:
    """Net present value for annual cashflows (year 0..N)."""
    r = float(discount_rate or 0.0)
    return float(sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cashflows)))


def _irr_bisection(cashflows: list[float], low: float = -0.95, high: float = 5.0, tol: float = 1e-7, max_iter: int = 200) -> float | None:
    """IRR via bisection on annual cashflows. Returns None if no sign change."""
    if not cashflows:
        return None

    def f(rate: float) -> float:
        # NPV at a given rate
        return float(sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows)))

    try:
        f_low = f(low)
        f_high = f(high)
    except Exception:
        return None

    # Need a sign change for bisection
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0:
        return None

    a, b = low, high
    fa, fb = f_low, f_high

    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = f(mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    return (a + b) / 2.0


def _battery_cycle_metrics(
    battery_cycles_equiv: float,
    days: float,
    assumed_life_years: float,
    cycle_life_efc: float,
    stress_ratio_threshold: float = 0.80,
) -> dict:
    """Derive cycle stress indicators from simulated equivalent full cycles."""
    cyc = max(float(battery_cycles_equiv or 0.0), 0.0)
    d = max(float(days or 0.0), 0.0)
    life = max(float(assumed_life_years or 0.0), 0.0)
    cycle_life = max(float(cycle_life_efc or 0.0), 0.0)
    efc_per_year = (cyc * (365.0 / d)) if d > 0 else 0.0
    implied_life = (cycle_life / efc_per_year) if (cycle_life > 0 and efc_per_year > 0) else None
    is_stress = bool(
        implied_life is not None
        and life > 0
        and implied_life < (life * float(stress_ratio_threshold or 0.80))
    )
    return {
        "efc_per_year": float(efc_per_year),
        "implied_cycle_life_years": (float(implied_life) if implied_life is not None else None),
        "is_cycle_stress": is_stress,
    }


def _build_battery_cashflows(
    annual_savings: float,
    batt_cost: float,
    battery_life_years: float,
    degradation_rate: float,
    price_growth_rate: float,
    cycle_aware: bool,
    efc_per_year: float,
    cycle_life_efc: float,
    eol_capacity_frac: float,
) -> dict:
    """Build annual battery cashflows, optionally cycle-aware with cycle-limited life."""
    base_savings = float(annual_savings or 0.0)
    life = max(float(battery_life_years or 0.0), 0.0)
    deg = max(float(degradation_rate or 0.0), 0.0)
    growth = float(price_growth_rate or 0.0)
    efc_y = max(float(efc_per_year or 0.0), 0.0)
    cycle_life = max(float(cycle_life_efc or 0.0), 0.0)
    eol_frac = max(min(float(eol_capacity_frac or 1.0), 1.0), 0.0)

    implied_cycle_life = None
    effective_life = life
    if cycle_aware and efc_y > 0 and cycle_life > 0:
        implied_cycle_life = cycle_life / efc_y
        effective_life = min(life, implied_cycle_life)

    effective_life = max(float(effective_life), 0.0)
    full_years = int(math.floor(effective_life + 1e-9))
    partial_year = max(min(effective_life - float(full_years), 1.0), 0.0)

    def _year_savings(yr: int) -> float:
        cal_mult = (1.0 - deg) ** (yr - 1)
        growth_mult = (1.0 + growth) ** (yr - 1)
        cycle_mult = 1.0
        if cycle_aware and efc_y > 0 and cycle_life > 0:
            cum_efc = min(efc_y * max(yr - 1, 0), cycle_life)
            wear = (cum_efc / cycle_life) if cycle_life > 0 else 0.0
            cycle_mult = max(eol_frac, 1.0 - ((1.0 - eol_frac) * wear))
        return float(base_savings * cal_mult * growth_mult * cycle_mult)

    savings_by_year: list[float] = []
    year_fractions: list[float] = []
    cashflows: list[float] = [-float(batt_cost or 0.0)]
    for yr in range(1, full_years + 1):
        sav = _year_savings(yr)
        savings_by_year.append(float(sav))
        year_fractions.append(1.0)
        cashflows.append(float(sav))
    if partial_year > 1e-9:
        sav_partial = _year_savings(full_years + 1) * partial_year
        savings_by_year.append(float(sav_partial))
        year_fractions.append(float(partial_year))
        cashflows.append(float(sav_partial))

    return {
        "cashflows": cashflows,
        "savings_by_year": savings_by_year,
        "year_fractions": year_fractions,
        "effective_life_years": float(effective_life),
        "implied_cycle_life_years": (float(implied_cycle_life) if implied_cycle_life is not None else None),
    }


def _import_kwh_series(df_intervals: pd.DataFrame) -> pd.Series:
    """Total import (general + controlled) as a Series."""
    if df_intervals is None or len(df_intervals) == 0:
        return pd.Series(dtype=float)
    g = df_intervals["general_kwh"] if "general_kwh" in df_intervals.columns else 0.0
    c = df_intervals["controlled_kwh"] if "controlled_kwh" in df_intervals.columns else 0.0
    return (g + c).astype(float)


def _band_name_for_ts(ts: pd.Timestamp, tou: TouTariff | None) -> str | None:
    """Return the TOU band name for a timestamp, or None."""
    if tou is None or not getattr(tou, "bands", None):
        return None
    for b in tou.bands:
        try:
            if _band_matches(ts, b):
                return str(b.name or "").strip()
        except Exception:
            continue
    return None


def _default_discharge_threshold_cents(plan: Plan) -> float | None:
    """Choose a sensible default discharge threshold (c/kWh) for Level 1.

    Meaning: only discharge when the current import rate is >= this threshold.
    - Flat plans: return None (discharge whenever importing).
    - TOU plans: default to SHOULDER (more homeowner-friendly), falling back to PEAK,
      then to the highest-rate band if neither is present.

    We prefer SHOULDER because many solar households still benefit from discharging
    during shoulder periods (avoided import cost often exceeds FiT forgone), whereas
    peak-only can understate savings and make the battery look idle.
    """
    if plan.import_type != "tou" or not plan.tou or not plan.tou.bands:
        return None

    rates: list[float] = []
    shoulder_rates: list[float] = []
    peak_rates: list[float] = []

    for b in plan.tou.bands:
        try:
            r = float(b.cents_per_kwh or 0.0)
        except Exception:
            r = 0.0
        rates.append(r)

        nm = str(getattr(b, "name", "") or "").strip().lower()
        if "shoulder" in nm:
            shoulder_rates.append(r)
        if "peak" in nm:
            peak_rates.append(r)

    # Default: lowest shoulder rate (so it discharges through all shoulder+peak)
    if shoulder_rates:
        return min(shoulder_rates)

    # Fallback: peak rate (lowest peak, if multiple)
    if peak_rates:
        return min(peak_rates)

    # Final fallback: highest band rate
    if rates:
        return max(rates)

    return None

def _intervals_wide_from_long(df_int: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format intervals (register,timestamp,kwh) into wide per-timestamp totals.

    Output columns: timestamp, general_kwh, controlled_kwh, export_kwh
    """
    if df_int is None or df_int.empty:
        return pd.DataFrame(columns=["timestamp","general_kwh","controlled_kwh","export_kwh"])

    d = df_int.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["register"] = d["register"].astype(str)

    g = d[d["register"].isin(GENERAL_REGS)].groupby("timestamp")["kwh"].sum()
    c = d[d["register"].isin(CONTROLLED_REGS)].groupby("timestamp")["kwh"].sum()
    e = d[d["register"].isin(EXPORT_REGS)].groupby("timestamp")["kwh"].sum()

    ts = sorted(set(g.index) | set(c.index) | set(e.index))
    out = pd.DataFrame({"timestamp": ts})
    out["general_kwh"] = out["timestamp"].map(g).fillna(0.0).astype(float)
    out["controlled_kwh"] = out["timestamp"].map(c).fillna(0.0).astype(float)
    out["export_kwh"] = out["timestamp"].map(e).fillna(0.0).astype(float)
    return out.sort_values("timestamp").reset_index(drop=True)


def _intervals_long_from_wide(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide per-timestamp totals back into long-format intervals.

    Produces three registers:
      - GENERAL_REGS[0] (E1)
      - CONTROLLED_REGS[0] (E2)
      - EXPORT_REGS[0] (B1)
    """
    if df_wide is None or df_wide.empty:
        return pd.DataFrame(columns=["register","timestamp","kwh"])

    d = df_wide.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    e1 = GENERAL_REGS[0] if GENERAL_REGS else "E1"
    e2 = CONTROLLED_REGS[0] if CONTROLLED_REGS else "E2"
    b1 = EXPORT_REGS[0] if EXPORT_REGS else "B1"

    parts = []
    for reg, col in [(e1, "general_kwh"), (e2, "controlled_kwh"), (b1, "export_kwh")]:
        if col not in d.columns:
            continue
        p = d[["timestamp", col]].copy()
        p = p.rename(columns={col: "kwh"})
        p["register"] = reg
        parts.append(p[["register","timestamp","kwh"]])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["register","timestamp","kwh"])
    out["kwh"] = pd.to_numeric(out["kwh"], errors="coerce").fillna(0.0).astype(float)
    return out.sort_values(["timestamp","register"]).reset_index(drop=True)


def apply_pv_scale_to_intervals(
    df_int: pd.DataFrame,
    solar_profile: Optional[pd.DataFrame],
    pv_scale: float,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict]:
    """Scale solar production and rebalance interval imports/exports.

    Returns:
      - adjusted long-format intervals (register,timestamp,kwh)
      - scaled solar production data (timestamp,pv_kwh) aligned to interval timestamps
      - metadata dict for diagnostics
    """
    meta = {
        "applied": False,
        "pv_scale": float(pv_scale or 0.0),
        "base_pv_kwh": 0.0,
        "scaled_pv_kwh": 0.0,
        "base_import_kwh": 0.0,
        "scaled_import_kwh": 0.0,
        "base_export_kwh": 0.0,
        "scaled_export_kwh": 0.0,
        "reason": "",
    }

    if df_int is None or not isinstance(df_int, pd.DataFrame) or df_int.empty:
        meta["reason"] = "No interval data."
        return pd.DataFrame(columns=["register", "timestamp", "kwh"]), None, meta

    if (
        not isinstance(solar_profile, pd.DataFrame)
        or solar_profile.empty
        or not {"timestamp", "pv_kwh"}.issubset(solar_profile.columns)
    ):
        meta["reason"] = "No aligned solar production data."
        return df_int.copy(), None, meta

    scale = max(float(pv_scale or 0.0), 0.0)

    wide = _intervals_wide_from_long(df_int)
    if wide.empty:
        meta["reason"] = "Could not build interval-wide profile."
        return df_int.copy(), None, meta

    s = solar_profile[["timestamp", "pv_kwh"]].copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
    s["pv_kwh"] = pd.to_numeric(s["pv_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    s = s.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False)["pv_kwh"].sum()

    wide = wide.copy()
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], errors="coerce")
    wide = wide.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for c in ("general_kwh", "controlled_kwh", "export_kwh"):
        if c not in wide.columns:
            wide[c] = 0.0
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)

    m = wide.merge(s, on="timestamp", how="left")
    m["pv_kwh"] = pd.to_numeric(m["pv_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

    m["import_kwh"] = m["general_kwh"] + m["controlled_kwh"]
    m["load_est_kwh"] = (m["import_kwh"] + m["pv_kwh"] - m["export_kwh"]).clip(lower=0.0)
    m["pv_scaled_kwh"] = m["pv_kwh"] * scale

    m["import_scaled_kwh"] = (m["load_est_kwh"] - m["pv_scaled_kwh"]).clip(lower=0.0)
    m["export_scaled_kwh"] = (m["pv_scaled_kwh"] - m["load_est_kwh"]).clip(lower=0.0)

    import_base = m["import_kwh"]
    controlled_share = (m["controlled_kwh"] / import_base.where(import_base > 0, float("nan"))).fillna(0.0).clip(lower=0.0, upper=1.0)
    m["controlled_scaled_kwh"] = m["import_scaled_kwh"] * controlled_share
    m["general_scaled_kwh"] = m["import_scaled_kwh"] - m["controlled_scaled_kwh"]

    wide_adj = pd.DataFrame({
        "timestamp": m["timestamp"],
        "general_kwh": m["general_scaled_kwh"].astype(float),
        "controlled_kwh": m["controlled_scaled_kwh"].astype(float),
        "export_kwh": m["export_scaled_kwh"].astype(float),
    })

    mapped_regs = set(GENERAL_REGS + CONTROLLED_REGS + EXPORT_REGS)
    base = df_int.copy()
    base["register"] = base["register"].astype(str)
    other_regs = base[~base["register"].isin(mapped_regs)].copy()

    adj_mapped = _intervals_long_from_wide(wide_adj)
    existing_mapped = set(base["register"]).intersection(mapped_regs)
    if float(wide_adj["general_kwh"].sum()) > 0 and GENERAL_REGS:
        existing_mapped.add(GENERAL_REGS[0])
    if float(wide_adj["controlled_kwh"].sum()) > 0 and CONTROLLED_REGS:
        existing_mapped.add(CONTROLLED_REGS[0])
    if float(wide_adj["export_kwh"].sum()) > 0 and EXPORT_REGS:
        existing_mapped.add(EXPORT_REGS[0])
    if existing_mapped:
        adj_mapped = adj_mapped[adj_mapped["register"].astype(str).isin(existing_mapped)].copy()

    out = pd.concat([other_regs, adj_mapped], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["kwh"] = pd.to_numeric(out["kwh"], errors="coerce").fillna(0.0).astype(float)
    out = out.dropna(subset=["timestamp"]).sort_values(["timestamp", "register"]).reset_index(drop=True)

    scaled_solar = m[["timestamp", "pv_scaled_kwh"]].rename(columns={"pv_scaled_kwh": "pv_kwh"}).copy()
    scaled_solar["pv_kwh"] = pd.to_numeric(scaled_solar["pv_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

    meta.update({
        "applied": True,
        "base_pv_kwh": float(m["pv_kwh"].sum()),
        "scaled_pv_kwh": float(scaled_solar["pv_kwh"].sum()),
        "base_import_kwh": float(m["import_kwh"].sum()),
        "scaled_import_kwh": float(m["import_scaled_kwh"].sum()),
        "base_export_kwh": float(m["export_kwh"].sum()),
        "scaled_export_kwh": float(m["export_scaled_kwh"].sum()),
        "reason": "",
    })
    return out, scaled_solar, meta


@dataclass
class EVParams:
    """User-configurable EV charging assumptions for interval-level what-if analysis."""

    enabled: bool = False
    annual_km: float = 12000.0
    consumption_kwh_per_100km: float = 17.0
    charging_loss_frac: float = 0.10
    charger_power_kw: float = 7.0
    charge_days: str = "all"  # all | wkday | wkend
    strategy: str = "timer_grid"  # timer_grid | solar_first_backup
    timer_start: str = "00:00"
    timer_end: str = "06:00"
    solar_start: str = "10:00"
    solar_end: str = "15:00"
    backup_start: str = "00:00"
    backup_end: str = "06:00"


def _infer_interval_minutes_from_timestamps(ts: pd.Series) -> float:
    """Infer interval length from timestamps; defaults to 5 minutes."""
    if ts is None or len(ts) < 2:
        return 5.0
    s = pd.to_datetime(ts, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(s) < 2:
        return 5.0
    mins = float((s.diff().dropna().dt.total_seconds() / 60.0).median())
    if not mins or pd.isna(mins) or mins <= 0:
        return 5.0
    return mins


def _ev_time_window_mask(ts: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    """Return mask for half-open window [start, end), supporting windows crossing midnight."""
    start_t = _parse_hhmm(str(start_hhmm or ""))
    end_t = _parse_hhmm(str(end_hhmm or ""))
    if start_t is None or end_t is None:
        return pd.Series(False, index=ts.index)

    start_m = int(start_t.hour) * 60 + int(start_t.minute)
    end_m = int(end_t.hour) * 60 + int(end_t.minute)
    tod_m = ts.dt.hour.astype(int) * 60 + ts.dt.minute.astype(int)

    if start_m == end_m:
        return pd.Series(True, index=ts.index)
    if start_m < end_m:
        return (tod_m >= start_m) & (tod_m < end_m)
    return (tod_m >= start_m) | (tod_m < end_m)


def _ev_day_filter_mask(ts: pd.Series, charge_days: str) -> pd.Series:
    """Return mask for selected charging day scope."""
    mode = str(charge_days or "all").strip().lower()
    if mode == "wkday":
        return ts.dt.weekday < 5
    if mode == "wkend":
        return ts.dt.weekday >= 5
    return pd.Series(True, index=ts.index)


def apply_ev_profile_to_intervals(df_int: pd.DataFrame, ev: EVParams) -> tuple[pd.DataFrame, dict]:
    """Apply EV charging to intervals and return (adjusted_df, summary)."""
    summary = {
        "enabled": bool(getattr(ev, "enabled", False)),
        "strategy": str(getattr(ev, "strategy", "timer_grid")),
        "annual_km": float(getattr(ev, "annual_km", 0.0) or 0.0),
        "consumption_kwh_per_100km": float(getattr(ev, "consumption_kwh_per_100km", 0.0) or 0.0),
        "charging_loss_frac": float(getattr(ev, "charging_loss_frac", 0.0) or 0.0),
        "charger_power_kw": float(getattr(ev, "charger_power_kw", 0.0) or 0.0),
        "interval_minutes": 5.0,
        "target_daily_kwh": 0.0,
        "requested_kwh": 0.0,
        "delivered_kwh": 0.0,
        "grid_kwh": 0.0,
        "solar_diverted_kwh": 0.0,
        "unmet_kwh": 0.0,
        "coverage_pct": 100.0,
        "active_days": 0,
        "days_with_charging": 0,
        "notes": "",
    }

    if df_int is None or not isinstance(df_int, pd.DataFrame) or df_int.empty:
        summary["notes"] = "No interval data available."
        return pd.DataFrame(columns=["register", "timestamp", "kwh"]), summary

    base = df_int.copy()
    if not bool(ev.enabled):
        return base, summary

    wide = _intervals_wide_from_long(base)
    if wide.empty:
        summary["notes"] = "Could not build mapped interval profile (E1/E2/B1)."
        return base, summary

    wide = wide.sort_values("timestamp").reset_index(drop=True)
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], errors="coerce")
    wide = wide.dropna(subset=["timestamp"]).reset_index(drop=True)
    if wide.empty:
        summary["notes"] = "No valid timestamps after parsing."
        return base, summary

    for c in ("general_kwh", "controlled_kwh", "export_kwh"):
        if c not in wide.columns:
            wide[c] = 0.0
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0).astype(float)
    wide["export_kwh"] = wide["export_kwh"].clip(lower=0.0)

    ts = pd.to_datetime(wide["timestamp"], errors="coerce")
    interval_minutes = _infer_interval_minutes_from_timestamps(ts)
    summary["interval_minutes"] = float(interval_minutes)

    annual_drive_kwh = max(float(ev.annual_km), 0.0) * max(float(ev.consumption_kwh_per_100km), 0.0) / 100.0
    loss = min(max(float(ev.charging_loss_frac), 0.0), 0.60)
    annual_wall_kwh = annual_drive_kwh / max(1e-9, (1.0 - loss))
    daily_target_kwh = annual_wall_kwh / 365.25 if annual_wall_kwh > 0 else 0.0
    summary["target_daily_kwh"] = float(daily_target_kwh)

    charge_day_mask = _ev_day_filter_mask(ts, ev.charge_days)
    date_vals = ts.dt.normalize()
    active_dates = sorted(pd.to_datetime(date_vals[charge_day_mask]).dropna().unique().tolist())
    summary["active_days"] = int(len(active_dates))

    if daily_target_kwh <= 0 or not active_dates:
        summary["notes"] = "EV load target is zero or no active charging days in dataset."
        return base, summary

    max_interval_kwh = max(float(ev.charger_power_kw), 0.0) * (float(interval_minutes) / 60.0)
    if max_interval_kwh <= 0:
        summary["notes"] = "Charger power must be greater than zero."
        return base, summary

    day_positions: dict[pd.Timestamp, list[int]] = {}
    for i, d in enumerate(date_vals):
        if pd.isna(d):
            continue
        day_positions.setdefault(pd.Timestamp(d), []).append(i)

    grid_add = [0.0] * len(wide)
    solar_divert = [0.0] * len(wide)
    export_vals = wide["export_kwh"].tolist()

    days_with_charging = 0
    requested_total = 0.0
    delivered_total = 0.0

    def _alloc_grid(positions: list[int], remaining: float) -> float:
        rem = float(remaining)
        for pos in positions:
            if rem <= 1e-9:
                break
            add_kwh = min(max_interval_kwh, rem)
            if add_kwh > 0:
                grid_add[pos] += add_kwh
                rem -= add_kwh
        return rem

    for day in active_dates:
        pos = day_positions.get(pd.Timestamp(day), [])
        if not pos:
            continue

        day_ts = ts.iloc[pos]
        remaining = float(daily_target_kwh)
        requested_total += float(daily_target_kwh)

        if str(ev.strategy) == "solar_first_backup":
            solar_mask = _ev_time_window_mask(day_ts, ev.solar_start, ev.solar_end).tolist()
            solar_pos = [p for p, keep in zip(pos, solar_mask) if keep]
            for p in solar_pos:
                if remaining <= 1e-9:
                    break
                divert = min(float(export_vals[p]), max_interval_kwh, remaining)
                if divert > 0:
                    export_vals[p] = max(float(export_vals[p]) - divert, 0.0)
                    solar_divert[p] += divert
                    remaining -= divert

            backup_mask = _ev_time_window_mask(day_ts, ev.backup_start, ev.backup_end).tolist()
            backup_pos = [p for p, keep in zip(pos, backup_mask) if keep]
            remaining = _alloc_grid(backup_pos, remaining)
        else:
            timer_mask = _ev_time_window_mask(day_ts, ev.timer_start, ev.timer_end).tolist()
            timer_pos = [p for p, keep in zip(pos, timer_mask) if keep]
            remaining = _alloc_grid(timer_pos, remaining)

        delivered_day = max(float(daily_target_kwh) - remaining, 0.0)
        delivered_total += delivered_day
        if delivered_day > 1e-6:
            days_with_charging += 1

    wide["general_kwh"] = wide["general_kwh"] + pd.Series(grid_add)
    wide["export_kwh"] = pd.Series(export_vals).clip(lower=0.0)

    mapped_regs = set(GENERAL_REGS + CONTROLLED_REGS + EXPORT_REGS)
    other_regs = base[~base["register"].astype(str).isin(mapped_regs)].copy()
    adj_mapped = _intervals_long_from_wide(wide)

    existing_mapped = set(base["register"].astype(str)).intersection(mapped_regs)
    if sum(grid_add) > 0 and GENERAL_REGS:
        existing_mapped.add(GENERAL_REGS[0])
    if sum(solar_divert) > 0 and EXPORT_REGS:
        existing_mapped.add(EXPORT_REGS[0])
    if existing_mapped:
        adj_mapped = adj_mapped[adj_mapped["register"].astype(str).isin(existing_mapped)].copy()

    out = pd.concat([other_regs, adj_mapped], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["kwh"] = pd.to_numeric(out["kwh"], errors="coerce").fillna(0.0).astype(float)
    out = out.dropna(subset=["timestamp"]).sort_values(["timestamp", "register"]).reset_index(drop=True)

    grid_total = float(sum(grid_add))
    solar_total = float(sum(solar_divert))
    delivered_total = float(grid_total + solar_total)
    unmet_total = max(float(requested_total) - delivered_total, 0.0)
    coverage = (delivered_total / float(requested_total) * 100.0) if requested_total > 0 else 100.0

    summary.update(
        {
            "requested_kwh": float(requested_total),
            "delivered_kwh": float(delivered_total),
            "grid_kwh": float(grid_total),
            "solar_diverted_kwh": float(solar_total),
            "unmet_kwh": float(unmet_total),
            "coverage_pct": float(coverage),
            "days_with_charging": int(days_with_charging),
            "notes": "",
        }
    )
    return out, summary


def _dataset_signature(df_int: pd.DataFrame) -> dict:
    """Small deterministic summary used to invalidate stale cached UI outputs."""
    sig = {
        "rows": 0,
        "ts_start": "",
        "ts_end": "",
        "general_kwh": 0.0,
        "controlled_kwh": 0.0,
        "export_kwh": 0.0,
    }
    if df_int is None or not isinstance(df_int, pd.DataFrame) or df_int.empty:
        return sig

    d = df_int.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["kwh"] = pd.to_numeric(d["kwh"], errors="coerce").fillna(0.0).astype(float)
    d["register"] = d["register"].astype(str)

    ts = d["timestamp"].dropna()
    sig["rows"] = int(len(d))
    sig["ts_start"] = str(ts.min()) if not ts.empty else ""
    sig["ts_end"] = str(ts.max()) if not ts.empty else ""
    sig["general_kwh"] = round(float(d.loc[d["register"].isin(GENERAL_REGS), "kwh"].sum()), 3)
    sig["controlled_kwh"] = round(float(d.loc[d["register"].isin(CONTROLLED_REGS), "kwh"].sum()), 3)
    sig["export_kwh"] = round(float(d.loc[d["register"].isin(EXPORT_REGS), "kwh"].sum()), 3)
    return sig

    rates: list[float] = []
    peak_like: list[float] = []

    for b in plan.tou.bands:
        try:
            r = float(b.cents_per_kwh or 0.0)
        except Exception:
            r = 0.0
        rates.append(r)
        nm = str(b.name or "").lower()
        if "peak" in nm:
            peak_like.append(r)

    if peak_like:
        # Discharge when current rate >= (a) peak rate
        return min(peak_like)
    if rates:
        return max(rates)
    return None


def apply_battery_to_intervals(
    df_int: pd.DataFrame,
    plan: Plan,
    batt: BatteryParams,
    solar_profile: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Adjust interval imports/exports after running the battery model.

    Expected input format (wide intervals):
      - timestamp (datetime-like)
      - general_kwh
      - controlled_kwh
      - export_kwh
    Optional solar production file format:
      - timestamp
      - pv_kwh (interval energy)

    Returns a copy with the same columns adjusted, plus:
      - battery_charge_kwh (kWh drawn from PV surplus)
      - battery_discharge_kwh (kWh delivered to loads)
      - soc_kwh (end-of-interval state of charge)
    """
    if batt is None or float(batt.capacity_kwh or 0.0) <= 0:
        return df_int.copy()

    df = df_int.copy()
    if "timestamp" not in df.columns:
        return df

    # Ensure required columns exist
    for c in ("general_kwh", "controlled_kwh", "export_kwh"):
        if c not in df.columns:
            df[c] = 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    timestamps = df["timestamp"].tolist()
    general = pd.to_numeric(df["general_kwh"], errors="coerce").fillna(0.0).astype(float).tolist()
    controlled = pd.to_numeric(df["controlled_kwh"], errors="coerce").fillna(0.0).astype(float).tolist()
    export = pd.to_numeric(df["export_kwh"], errors="coerce").fillna(0.0).astype(float).tolist()

    pv_aligned = None
    if (
        isinstance(solar_profile, pd.DataFrame)
        and not solar_profile.empty
        and {"timestamp", "pv_kwh"}.issubset(solar_profile.columns)
    ):
        pv_df = solar_profile.copy()
        pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"], errors="coerce")
        pv_df["pv_kwh"] = pd.to_numeric(pv_df["pv_kwh"], errors="coerce").fillna(0.0).astype(float)
        pv_df = pv_df.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False)["pv_kwh"].sum()
        pv_map = pv_df.set_index("timestamp")["pv_kwh"]
        pv_aligned = df["timestamp"].map(pv_map).fillna(0.0).astype(float).tolist()

    charge_vals = [0.0] * n
    discharge_vals = [0.0] * n
    soc_vals = [0.0] * n

    # Interval length (hours)
    if len(df) >= 2:
        dt_s = df["timestamp"].diff().dt.total_seconds().median()
        dt_hours = (dt_s / 3600.0) if pd.notna(dt_s) and dt_s > 0 else (5.0 / 60.0)
    else:
        dt_hours = 5.0 / 60.0

    cap = float(batt.capacity_kwh)
    power = float(batt.power_kw or 0.0)
    power = max(power, 0.0)
    charge_eff, discharge_eff = _split_roundtrip_eff(batt.roundtrip_eff)

    reserve_kwh = cap * float(batt.reserve_frac or 0.0)
    reserve_kwh = max(min(reserve_kwh, cap), 0.0)

    soc = cap * float(batt.initial_soc_frac or 0.0)
    soc = max(min(soc, cap), 0.0)

    discharge_min = batt.discharge_min_rate_cents
    if discharge_min is None:
        discharge_min = _default_discharge_threshold_cents(plan)

    max_energy_per_interval = power * dt_hours  # kWh (both charge & discharge limit)
    discharge_threshold = float(discharge_min) if discharge_min is not None else None
    tou_rates = None
    flat_rate = None
    if discharge_threshold is not None:
        if plan.import_type == "tou":
            tou_rates = [tou_rate_for_ts(pd.Timestamp(ts), plan.tou) for ts in timestamps]
        else:
            flat_rate = float(plan.flat.cents_per_kwh) if plan.flat else 0.0

    for i in range(n):
        ts = pd.Timestamp(timestamps[i])

        g = float(general[i] or 0.0)
        c = float(controlled[i] or 0.0)
        e = float(export[i] or 0.0)
        pv = float(pv_aligned[i] or 0.0) if pv_aligned is not None else None

        charge_kwh = 0.0      # from PV surplus (kWh)
        discharge_kwh = 0.0   # delivered to loads (kWh)

        # If solar production is supplied, derive an interval load estimate and route PV first.
        if pv is not None:
            pv = max(pv, 0.0)
            # Estimated total site load from import/export/PV balance
            load_total = max((g + c) + pv - e, 0.0)
            # Approximate controlled load as metered controlled import
            c_load = max(min(c, load_total), 0.0)
            g_load = max(load_total - c_load, 0.0)

            # PV offsets general load first; controlled remains as separate circuit assumption.
            pv_to_g = min(pv, g_load)
            g = g_load - pv_to_g
            c = c_load
            pv_surplus = max(pv - pv_to_g, 0.0)

            # 1) Charge from PV surplus only (grid charging OFF)
            if batt.charge_from_export_only and pv_surplus > 0 and soc < cap and max_energy_per_interval > 0:
                room = cap - soc
                energy_from_pv = min(pv_surplus, max_energy_per_interval, room / max(charge_eff, 1e-9))
                if energy_from_pv > 0:
                    soc += energy_from_pv * charge_eff
                    charge_kwh += energy_from_pv
                    pv_surplus -= energy_from_pv
            e = max(pv_surplus, 0.0)
        else:
            # 1) Charge from export only (grid charging OFF)
            if batt.charge_from_export_only and e > 0 and soc < cap and max_energy_per_interval > 0:
                room = cap - soc
                # Need energy_from_pv such that (energy_from_pv * charge_eff) <= room
                energy_from_pv = min(e, max_energy_per_interval, room / max(charge_eff, 1e-9))
                if energy_from_pv > 0:
                    soc += energy_from_pv * charge_eff
                    charge_kwh += energy_from_pv
                    e -= energy_from_pv

        # 2) Discharge to offset imports (prioritise general then controlled)
        import_total = g + c
        if import_total > 0 and soc > reserve_kwh and max_energy_per_interval > 0:
            allow_discharge = True
            if discharge_threshold is not None:
                # Current import rate (c/kWh)
                if tou_rates is not None:
                    r = tou_rates[i]
                    allow_discharge = (r is not None) and (float(r) >= discharge_threshold)
                else:
                    allow_discharge = float(flat_rate or 0.0) >= discharge_threshold

            if allow_discharge:
                # Energy available to loads (accounting for discharge efficiency)
                available_to_load = (soc - reserve_kwh) * discharge_eff
                energy_to_load = min(import_total, max_energy_per_interval, available_to_load)
                if energy_to_load > 0:
                    soc -= energy_to_load / max(discharge_eff, 1e-9)
                    discharge_kwh += energy_to_load

                    # Reduce general first, then controlled
                    red_g = min(g, energy_to_load)
                    g -= red_g
                    remaining = energy_to_load - red_g
                    if remaining > 0:
                        red_c = min(c, remaining)
                        c -= red_c

        general[i] = g
        controlled[i] = c
        export[i] = e
        charge_vals[i] = charge_kwh
        discharge_vals[i] = discharge_kwh
        soc_vals[i] = soc

    df["general_kwh"] = general
    df["controlled_kwh"] = controlled
    df["export_kwh"] = export
    df["battery_charge_kwh"] = charge_vals
    df["battery_discharge_kwh"] = discharge_vals
    df["soc_kwh"] = soc_vals

    return df




def _colsum(df: pd.DataFrame, col: str) -> float:
    """Safe sum for a dataframe column; returns 0.0 if missing."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    except Exception:
        return 0.0


def simulate_plan_with_battery(
    df_int: pd.DataFrame,
    plan: Plan,
    batt: BatteryParams,
    baseline: Optional[dict] = None,
    wide_base: Optional[pd.DataFrame] = None,
    solar_profile: Optional[pd.DataFrame] = None,
) -> dict:
    """Run simulate_plan on baseline vs battery-adjusted intervals and attach $ impacts + battery KPIs.

    IMPORTANT: The app's main interval dataframe is long-format (register,timestamp,kwh).
    The battery dispatch operates on *per-timestamp totals* (wide format). We therefore:
      1) simulate baseline on long-format
      2) convert to wide, dispatch battery, convert back to long
      3) simulate adjusted on long-format

    One-time plan sign-up credits are excluded from this model so NPV/IRR are not distorted
    by non-recurring switching incentives.
    """
    # Baseline (no battery) can be injected by callers during sweeps to avoid recomputation.
    baseline_sim = baseline if baseline is not None else simulate_plan(df_int, plan, include_signup_credit=False)

    # Convert to wide totals, dispatch battery, then back to long for billing.
    # Reusing precomputed wide intervals is a major speed-up in optimizer loops.
    wide = wide_base.copy() if isinstance(wide_base, pd.DataFrame) else _intervals_wide_from_long(df_int)
    wide_adj = apply_battery_to_intervals(wide, plan, batt, solar_profile=solar_profile)
    df_adj_long = _intervals_long_from_wide(wide_adj)

    # Battery-adjusted bill simulation
    sim = simulate_plan(df_adj_long, plan, include_signup_credit=False)

    # Attach baseline totals for comparison
    sim["baseline_total_cents"] = float(baseline_sim.get("total_cents", 0.0))
    sim["baseline_general_cents"] = float(baseline_sim.get("general_cents", 0.0))
    sim["baseline_controlled_cents"] = float(baseline_sim.get("controlled_cents", 0.0))
    sim["baseline_export_credit_cents"] = float(baseline_sim.get("export_credit_cents", 0.0))
    sim["baseline_supply_cents"] = float(baseline_sim.get("supply_cents", 0.0))
    sim["baseline_controlled_supply_cents"] = float(baseline_sim.get("controlled_supply_cents", 0.0))
    sim["baseline_monthly_fee_cents"] = float(baseline_sim.get("monthly_fee_cents", 0.0))

    # $ impacts (positive = bill reduced / savings)
    sim["savings_total_cents"] = sim["baseline_total_cents"] - float(sim.get("total_cents", 0.0))

    baseline_import_cents = sim["baseline_general_cents"] + sim["baseline_controlled_cents"]
    batt_import_cents = float(sim.get("general_cents", 0.0)) + float(sim.get("controlled_cents", 0.0))
    sim["savings_import_cents"] = baseline_import_cents - batt_import_cents

    # Export credit change (usually negative because export drops)
    sim["fit_credit_change_cents"] = float(sim.get("export_credit_cents", 0.0)) - sim["baseline_export_credit_cents"]
    sim["lost_fit_cents"] = -sim["fit_credit_change_cents"]  # positive when FiT credit decreases

    sim["fixed_charge_change_cents"] = (
        (float(sim.get("supply_cents", 0.0)) + float(sim.get("controlled_supply_cents", 0.0)) + float(sim.get("monthly_fee_cents", 0.0)))
        - (sim["baseline_supply_cents"] + sim["baseline_controlled_supply_cents"] + sim["baseline_monthly_fee_cents"])
    )

    # Battery parameters
    sim["battery_capacity_kwh"] = float(batt.capacity_kwh)
    sim["battery_power_kw"] = float(batt.power_kw)
    sim["battery_roundtrip_eff"] = float(batt.roundtrip_eff)
    sim["battery_discharge_min_rate_cents"] = batt.discharge_min_rate_cents

    # Battery energy flows (over the simulated period) - these live on the wide dataframe
    sim["battery_charge_kwh"] = _colsum(wide_adj, "battery_charge_kwh")
    sim["battery_discharge_kwh"] = _colsum(wide_adj, "battery_discharge_kwh")
    sim["battery_cycles_equiv"] = (sim["battery_discharge_kwh"] / sim["battery_capacity_kwh"]) if sim["battery_capacity_kwh"] > 0 else 0.0

    # Baseline vs battery-adjusted import/export (kWh)
    sim["baseline_import_kwh"] = float(baseline_sim.get("general_kwh", 0.0)) + float(baseline_sim.get("controlled_kwh", 0.0))
    sim["baseline_export_kwh"] = float(baseline_sim.get("export_kwh", 0.0))
    sim["battery_import_kwh"] = _colsum(wide_adj, "general_kwh") + _colsum(wide_adj, "controlled_kwh")
    sim["battery_export_kwh"] = _colsum(wide_adj, "export_kwh")
    sim["import_reduction_kwh"] = sim["baseline_import_kwh"] - sim["battery_import_kwh"]
    sim["export_reduction_kwh"] = sim["baseline_export_kwh"] - sim["battery_export_kwh"]

    return sim


def daily_totals(df_intervals: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    gen = df[df["register"].isin(GENERAL_REGS)].groupby("date")["kwh"].sum()
    ctl = df[df["register"].isin(CONTROLLED_REGS)].groupby("date")["kwh"].sum()
    exp = df[df["register"].isin(EXPORT_REGS)].groupby("date")["kwh"].sum()

    days = sorted(set(gen.index) | set(ctl.index) | set(exp.index))
    out = pd.DataFrame({"date": days})
    out["general_kwh"] = out["date"].map(gen).fillna(0.0)
    out["controlled_kwh"] = out["date"].map(ctl).fillna(0.0)
    out["export_kwh"] = out["date"].map(exp).fillna(0.0)

    out["supply_cents"] = float(plan.supply_cents_per_day or 0.0)
    out["controlled_supply_cents"] = float(plan.controlled_supply_cents_per_day or 0.0)
    controlled_rate_c = float(plan.controlled_cents_per_kwh or 0.0)

    if plan.import_type == "flat":
        rate = float(plan.flat.cents_per_kwh if plan.flat else 0.0)
        out["general_cents"] = out["general_kwh"] * rate
        if _flat_plan_uses_single_import_rate(plan):
            out["controlled_cents"] = out["controlled_kwh"] * rate
        else:
            out["controlled_cents"] = out["controlled_kwh"] * controlled_rate_c
    else:
        dfg = df[df["register"].isin(GENERAL_REGS)].copy()
        dfg = dfg.groupby("timestamp", as_index=False)["kwh"].sum()
        dfg["date"] = dfg["timestamp"].dt.date
        dfg["rate"] = dfg["timestamp"].apply(lambda ts: tou_rate_for_ts(pd.Timestamp(ts), plan.tou))
        dfg["cents"] = dfg["kwh"] * dfg["rate"]
        gen_day_cents = dfg.groupby("date")["cents"].sum()
        out["general_cents"] = out["date"].map(gen_day_cents).fillna(0.0)
        out["controlled_cents"] = out["controlled_kwh"] * controlled_rate_c

    if _has_fit_tou(plan):
        dfe = df[df["register"].isin(EXPORT_REGS)].copy()
        dfe = dfe.groupby("timestamp", as_index=False)["kwh"].sum()
        dfe["date"] = dfe["timestamp"].dt.date
        dfe["rate"] = dfe["timestamp"].apply(lambda ts: fit_rate_for_ts(pd.Timestamp(ts), plan))
        dfe["cents"] = dfe["kwh"] * dfe["rate"]
        exp_day_cents = dfe.groupby("date")["cents"].sum()
        out["export_credit_cents"] = out["date"].map(exp_day_cents).fillna(0.0)
    elif plan.feed_in_tiered:
        sim = simulate_plan(df_intervals, plan, include_signup_credit=False)
        total_credit = float(sim["export_credit_cents"])
        total_export = float(sim["export_kwh"])
        out["export_credit_cents"] = (out["export_kwh"] / total_export * total_credit) if total_export > 0 else 0.0
    else:
        out["export_credit_cents"] = out["export_kwh"] * float(plan.feed_in_flat_cents_per_kwh or 0.0)

    out["total_cents"] = (
        out["supply_cents"]
        + out["controlled_supply_cents"]
        + out["general_cents"]
        + out["controlled_cents"]
        - out["export_credit_cents"]
    )
    out["total_$"] = out["total_cents"] / 100.0
    return out


def monthly_totals(df_intervals: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    """Monthly totals ($) derived from daily totals, for risk/volatility analysis."""
    d = daily_totals(df_intervals, plan).copy()
    if d.empty:
        return pd.DataFrame(columns=["month", "total_$"])
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    m = d.groupby("month", as_index=False)["total_$"].sum()
    return m.sort_values("month").reset_index(drop=True)


# ---------------------------
# Forecast helpers
# ---------------------------
def _last_n_days_slice(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    max_ts = df["timestamp"].max()
    min_ts = max_ts - timedelta(days=n_days)
    return df[df["timestamp"] > min_ts].copy()


def forecast_repeat_last_4_weeks(df_intervals: pd.DataFrame, days_forward: int) -> pd.DataFrame:
    base = _last_n_days_slice(df_intervals, 28)
    if base.empty:
        return pd.DataFrame(columns=df_intervals.columns)

    base["date"] = base["timestamp"].dt.date
    base_days = sorted(base["date"].unique().tolist())
    if not base_days:
        return pd.DataFrame(columns=df_intervals.columns)

    out_parts = []
    start_future = base["timestamp"].max().normalize() + timedelta(days=1)

    for d in range(days_forward):
        src_day = base_days[d % len(base_days)]
        day_slice = base[base["date"] == src_day].copy()
        target_day = (start_future + timedelta(days=d)).date()
        day_slice["timestamp"] = day_slice["timestamp"].apply(
            lambda ts: pd.Timestamp(dt.datetime.combine(target_day, ts.time()))
        )
        out_parts.append(day_slice.drop(columns=["date"]))

    return pd.concat(out_parts, ignore_index=True)


def forecast_replay_last_year(df_intervals: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df_intervals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days
    if span_days < 330:
        return None

    last_year_start = df["timestamp"].max() - timedelta(days=365)
    last_year = df[df["timestamp"] >= last_year_start].copy()
    last_year["timestamp"] = last_year["timestamp"] + timedelta(days=365)
    return last_year


def render_help_and_glossary() -> None:
    st.subheader("Help and glossary")
    st.write(
        "This guide explains what to upload, how results are calculated, and how to read the main outputs."
    )

    with st.expander("Quick start", expanded=True):
        st.markdown(
            "1. Upload your NEM12 CSV file.\n"
            "2. Optionally upload a solar production file (CSV/XLSX) for battery and solar diagnostics.\n"
            "3. Choose your current retailer in the sidebar.\n"
            "4. Review Comparison results first, then use Breakdowns for detailed checks.\n"
            "5. Use Plan Library to add or edit retailer plans."
        )

    with st.expander("Input files and register mapping", expanded=False):
        st.markdown(
            "- Required: NEM12 interval CSV containing 200/300 records.\n"
            "- Optional: Solar production file for generation alignment and battery realism.\n"
            "- Locked register mapping used by this app:\n"
            "  - `E1`: General import\n"
            "  - `E2`: Controlled load import\n"
            "  - `B1`: Export"
        )

    with st.expander("How to interpret key outputs", expanded=False):
        st.markdown(
            "- `Total ($)`: Estimated total bill over the dataset period (GST inclusive).\n"
            "- `Effective Rate (c/kWh)`: Total bill divided by total imported kWh.\n"
            "- `Avg import rate excl supply (c/kWh)`: Usage-only import rate net of FiT credit, excluding fixed daily charges.\n"
            "- `FiT credit ($)`: Credit for exported energy based on plan feed-in settings.\n"
            "- `Supply ($)`: Daily fixed charges over the period.\n"
            "- `Controlled usage ($)`: E2 usage charges.\n"
            "- `Max demand (30-min avg)`: Highest rolling 30-minute average import demand."
        )
        st.caption(
            "Sign-up credits may be included once in the comparison table totals, but are excluded in forecast and optimizer modelling."
        )

    with st.expander("Screen-by-screen guide", expanded=False):
        st.markdown(
            "- `Overview`: Dataset sanity checks, totals, and demand profile.\n"
            "- `Comparison`: Fast ranking of plans by estimated total cost.\n"
            "- `Breakdowns`: Daily totals, TOU bands, invoice-style line items, forecast, risk, sensitivity, battery outputs.\n"
            "- `Battery simulation`: Detailed battery economics and dispatch results.\n"
            "- `Plan Library`: Create, duplicate, edit, export, and import plans."
        )

    with st.expander("Operational rules (Plan Library on Streamlit Cloud)", expanded=False):
        st.markdown(
            "1. **Shared app library**: Plan Library changes are app-level on the running instance, not per-user accounts.\n"
            "2. **Non-durable storage**: Streamlit Cloud local files can reset on reboot/redeploy.\n"
            "3. **Default startup source**: If repo `plans.json` is missing/invalid, app starts with built-in defaults.\n"
            "4. **Best practice for users**: After edits, use `Download plans.json` as personal backup.\n"
            "5. **Restore workflow**: Use `Upload plans.json to replace library` to restore full plan sets.\n"
            "6. **Admin rule**: Keep the canonical `plans.json` committed to GitHub `main` for stable startup."
        )

    with st.expander("Glossary", expanded=False):
        st.markdown(
            "- `NEM12`: Australian interval metering format used for consumption/export data.\n"
            "- `TOU (Time of Use)`: Different import/export rates by time window and day type.\n"
            "- `FiT (Feed-in Tariff)`: Credit rate paid for exported energy.\n"
            "- `Controlled load`: Typically separately metered appliances (often hot water) billed at controlled load tariffs.\n"
            "- `Shoulder/Peak/Off-peak`: Named TOU rate periods.\n"
            "- `Self-consumption`: Solar generation used on-site rather than exported.\n"
            "- `Solar coverage`: Share of site load met by self-consumed solar.\n"
            "- `Demand (kW)`: Power level; in this app, derived from interval energy data."
        )

    with st.expander("Troubleshooting", expanded=False):
        st.markdown(
            "- `Could not parse any 200/300 interval data`: Confirm file is NEM12 and contains interval rows.\n"
            "- Missing expected registers: Check your meter/export channels in the source file.\n"
            "- Solar alignment is low: Confirm timestamps, timezone, and interval cadence match your NEM12 file.\n"
            "- Unexpected costs: Review plan settings in Plan Library (TOU windows, FiT, daily charges, credits)."
        )

    with st.expander("Download user documents", expanded=False):
        docs = [
            ("User Guide", BASE_DIR / "Energex_Compare_App_User_Guide.docx"),
            ("App Summary", BASE_DIR / "Energex_Compare_App_Summary.docx"),
        ]
        any_doc = False
        for label, path in docs:
            if path.exists():
                any_doc = True
                st.download_button(
                    f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"download_{path.stem}",
                )
        if not any_doc:
            st.caption("No local .docx help documents were found in the app folder.")


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Energex Retailer Comparator", layout='wide')

# --- Navigation (pro polish) ---
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Comparison", "Breakdowns", "Battery simulation", "Plan Library", "Help & Glossary"],
    index=0,
)
show_overview_only = section == "Overview"
show_comparison_only = section == "Comparison"
show_battery_only = section == "Battery simulation"
show_plan_library_only = section == "Plan Library"
show_help_only = section == "Help & Glossary"
st.title("Electricity plan comparator (Energex) - NEM12 5-minute CSV")

st.write(
    "Upload your **NEM12** CSV (5-minute intervals). This app compares plans using a locked mapping:\n"
    "- **E1** = general import\n"
    "- **E2** = controlled load\n"
    "- **B1** = export\n"
)
st.sidebar.caption("Need guidance? Open `Help & Glossary` in Navigate.")

if show_help_only:
    render_help_and_glossary()
    st.stop()

uploaded = st.file_uploader("Upload NEM12 CSV", type=["csv"])
if not uploaded:
    st.info("Upload a NEM12 CSV to begin.")
    st.stop()

df_int_base = read_nem12_5min(uploaded)
if df_int_base.empty:
    st.error("Could not parse any 200/300 interval data from the file.")
    st.stop()

uploaded_solar = st.file_uploader(
    "Optional: Upload solar production file (CSV/XLSX)",
    type=["csv", "xlsx", "xls"],
    key="solar_upload",
    help="Supports Fronius Solar.Web style exports. Used to improve battery dispatch realism and solar diagnostics.",
)

regs = sorted(df_int_base["register"].unique().tolist())
st.caption(f"Registers detected in file: {', '.join(regs)}")

# Load plan library into session
defaults = [ORIGIN, ALINTA]
if "plans_lib" not in st.session_state:
    st.session_state["plans_lib"] = load_plans(defaults)
if "battery_assumptions_cfg" not in st.session_state:
    st.session_state["battery_assumptions_cfg"] = load_battery_assumptions_config()

plans = st.session_state["plans_lib"]
battery_assumptions_cfg = st.session_state["battery_assumptions_cfg"]

# Back-compat alias (some sections refer to `plans_lib`)
plans_lib = plans

# Ensure at least defaults exist
if not plans:
    plans = defaults
    st.session_state["plans_lib"] = plans
plans_lib = plans

st.caption(f"Plans file: {PLANS_FILE}")
st.caption(f"Plans loaded: {len(plans)}")
if LAST_PLANS_LOAD_STATUS.get("source") != "file":
    reason = str(LAST_PLANS_LOAD_STATUS.get("reason", "unknown")).replace("_", " ")
    st.warning(
        "Using built-in default plans only. "
        f"Reason: {reason}. "
        "Add `plans.json` to your GitHub repo or import it in Plan Library."
    )
elif int(LAST_PLANS_LOAD_STATUS.get("skipped", 0) or 0) > 0:
    st.info(
        f"Loaded {LAST_PLANS_LOAD_STATUS.get('count', len(plans))} plans from file; "
        f"skipped {LAST_PLANS_LOAD_STATUS.get('skipped', 0)} invalid entries."
    )

if LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS.get("source") != "file":
    reason = str(LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS.get("reason", "unknown")).replace("_", " ")
    st.caption(
        f"Battery assumptions presets using built-in defaults ({reason}). "
        f"File path: {BATTERY_ASSUMPTIONS_FILE}"
    )
elif int(LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS.get("skipped", 0) or 0) > 0:
    st.caption(
        f"Battery assumptions loaded from file; "
        f"skipped {LAST_BATTERY_ASSUMPTIONS_LOAD_STATUS.get('skipped', 0)} invalid preset entries."
    )


# ---------------------------
# Current retailer (homeowner-friendly defaults)
# ---------------------------
plan_names_sidebar = [p.name for p in plans] if plans else []
pending_current_retailer = st.session_state.pop("_pending_current_retailer", None)
pending_invoice_plan = st.session_state.pop("_pending_invoice_plan", None)
if plan_names_sidebar:
    if isinstance(pending_current_retailer, str) and pending_current_retailer in plan_names_sidebar:
        st.session_state["current_retailer"] = pending_current_retailer
    if isinstance(pending_invoice_plan, str) and pending_invoice_plan in plan_names_sidebar:
        st.session_state["invoice_plan"] = pending_invoice_plan

    if ("current_retailer" not in st.session_state) or (st.session_state["current_retailer"] not in plan_names_sidebar):
        st.session_state["current_retailer"] = plan_names_sidebar[0]

    st.sidebar.markdown("### Your setup")
    st.sidebar.selectbox(
        "My current retailer",
        plan_names_sidebar,
        index=plan_names_sidebar.index(st.session_state["current_retailer"]),
        key="current_retailer",
        help="This plan will be selected by default across tabs and highlighted in results.",
    )
else:
    st.sidebar.markdown("### Your setup")

if "has_home_battery" not in st.session_state:
    try:
        st.session_state["has_home_battery"] = float(st.session_state.get("current_setup_battery_kwh", 0.0) or 0.0) > 0.0
    except Exception:
        st.session_state["has_home_battery"] = False

has_home_battery = st.sidebar.checkbox(
    "I currently have a home battery",
    value=bool(st.session_state.get("has_home_battery", False)),
    key="has_home_battery",
    help=(
        "Used to keep retailer ranking fair. If off, battery-dependent plans are excluded by default "
        "from Comparison results."
    ),
)
if has_home_battery:
    include_battery_dependent_in_comparison = True
    st.session_state["include_battery_dependent_in_comparison"] = False
else:
    include_battery_dependent_in_comparison = st.sidebar.checkbox(
        "Include battery-dependent plans in Comparison results",
        value=bool(st.session_state.get("include_battery_dependent_in_comparison", False)),
        key="include_battery_dependent_in_comparison",
        help="Turn on to force these plans back into the no-battery comparison table.",
    )

with st.sidebar.expander("EV charging scenario", expanded=False):
    ev_enabled = st.checkbox(
        "Include EV charging",
        value=False,
        key="ev_enabled",
        help="Adds EV charging demand to your interval profile so plan and battery outputs include EV impact.",
    )
    ev_annual_km = st.number_input(
        "Annual driving (km)",
        min_value=0.0,
        max_value=100000.0,
        value=12000.0,
        step=500.0,
        key="ev_annual_km",
        disabled=not ev_enabled,
        help="Enter your expected yearly driving distance. If unsure, 12,000 km/year is a common baseline.",
    )
    ev_consumption = st.number_input(
        "EV consumption (kWh/100km)",
        min_value=5.0,
        max_value=40.0,
        value=17.0,
        step=0.5,
        key="ev_consumption_kwh_per_100km",
        disabled=not ev_enabled,
        help="Vehicle energy use from the wall per 100 km. Typical range is about 14-22 kWh/100km.",
    )
    ev_loss_pct = st.number_input(
        "Charging losses (%)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        key="ev_charge_loss_pct",
        disabled=not ev_enabled,
        help="Charging/inverter losses between grid/solar and battery. If unsure, 8-12% is typical.",
    )
    ev_charger_kw = st.number_input(
        "Charger power (kW)",
        min_value=1.0,
        max_value=22.0,
        value=7.0,
        step=0.5,
        key="ev_charger_kw",
        disabled=not ev_enabled,
        help="Your home EV charger's maximum AC power. This is charger capability, not battery size.",
    )
    ev_charge_days_label = st.selectbox(
        "Charge on",
        ["All days", "Weekdays only", "Weekends only"],
        index=0,
        key="ev_charge_days",
        disabled=not ev_enabled,
        help="Select which days EV charging is allowed in the simulation.",
    )
    if st.session_state.get("ev_strategy") == "Solar-first + timer backup":
        st.session_state["ev_strategy"] = "Solar First + Timer Backup"
    ev_strategy_label = st.selectbox(
        "Charging strategy",
        ["Timer (grid charging)", "Solar First + Timer Backup"],
        index=1,
        key="ev_strategy",
        disabled=not ev_enabled,
        help="Timer uses configured time windows from grid. Solar First diverts export first, then uses backup timer if needed.",
    )

    ev_timer_start_t = st.time_input(
        "Timer start",
        value=dt.time(0, 0),
        key="ev_timer_start",
        disabled=not ev_enabled,
        help="Local clock time. This timer window can cross midnight.",
    )
    ev_timer_end_t = st.time_input(
        "Timer end",
        value=dt.time(6, 0),
        key="ev_timer_end",
        disabled=not ev_enabled,
        help="Local clock time. This timer window can cross midnight.",
    )

    if ev_strategy_label == "Solar First + Timer Backup":
        ev_solar_start_t = st.time_input(
            "Solar window start",
            value=dt.time(10, 0),
            key="ev_solar_start",
            disabled=not ev_enabled,
            help="Local clock time. EV charging can use exported solar only inside this window.",
        )
        ev_solar_end_t = st.time_input(
            "Solar window end",
            value=dt.time(15, 0),
            key="ev_solar_end",
            disabled=not ev_enabled,
            help="Local clock time. EV charging can use exported solar only inside this window.",
        )
        ev_backup_start_t = st.time_input(
            "Backup window start",
            value=dt.time(0, 0),
            key="ev_backup_start",
            disabled=not ev_enabled,
            help="If solar is not enough, grid charging is allowed in this backup window.",
        )
        ev_backup_end_t = st.time_input(
            "Backup window end",
            value=dt.time(6, 0),
            key="ev_backup_end",
            disabled=not ev_enabled,
            help="If solar is not enough, grid charging is allowed in this backup window.",
        )
    else:
        ev_solar_start_t = st.session_state.get("ev_solar_start", dt.time(10, 0))
        ev_solar_end_t = st.session_state.get("ev_solar_end", dt.time(15, 0))
        ev_backup_start_t = st.session_state.get("ev_backup_start", dt.time(0, 0))
        ev_backup_end_t = st.session_state.get("ev_backup_end", dt.time(6, 0))

    if ev_enabled:
        drive_kwh_yr = float(ev_annual_km) * float(ev_consumption) / 100.0
        wall_kwh_yr = drive_kwh_yr / max(1e-9, (1.0 - float(ev_loss_pct) / 100.0))
        st.caption(f"Estimated EV charging demand: {wall_kwh_yr:,.0f} kWh/yr (~{wall_kwh_yr/365.25:.1f} kWh/day).")

if "current_setup_solar_kw" not in st.session_state:
    st.session_state["current_setup_solar_kw"] = float(st.session_state.get("joint_current_pv_kw", 6.6))
default_current_setup_battery_kwh = 13.5 if bool(has_home_battery) else 0.0
if "current_setup_battery_kwh" not in st.session_state:
    st.session_state["current_setup_battery_kwh"] = float(default_current_setup_battery_kwh)
if "_last_has_home_battery_for_profile" not in st.session_state:
    st.session_state["_last_has_home_battery_for_profile"] = bool(has_home_battery)
elif bool(st.session_state.get("_last_has_home_battery_for_profile")) != bool(has_home_battery):
    try:
        current_kwh_profile = float(st.session_state.get("current_setup_battery_kwh", 0.0) or 0.0)
    except Exception:
        current_kwh_profile = 0.0
    if not bool(has_home_battery):
        # Apply 0 kWh as the default when turning battery off, but preserve custom overrides.
        if (current_kwh_profile <= 0.0) or (abs(current_kwh_profile - 13.5) < 1e-9):
            st.session_state["current_setup_battery_kwh"] = 0.0
    else:
        if current_kwh_profile <= 0.0:
            st.session_state["current_setup_battery_kwh"] = 13.5
    st.session_state["_last_has_home_battery_for_profile"] = bool(has_home_battery)
if "current_setup_battery_power_kw" not in st.session_state:
    st.session_state["current_setup_battery_power_kw"] = 5.0
if "current_setup_ev_enabled" not in st.session_state:
    st.session_state["current_setup_ev_enabled"] = bool(ev_enabled)
if "current_setup_ev_km_yr" not in st.session_state:
    st.session_state["current_setup_ev_km_yr"] = float(ev_annual_km)

with st.sidebar.expander("Current setup profile", expanded=False):
    st.caption(
        "Used to prefill What-if inputs (auto-updates when you change these values). "
        "Battery size defaults to 0 kWh when home battery is off, and can still be overridden."
    )
    st.number_input(
        "Current solar size (kW)",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state.get("current_setup_solar_kw", 6.6)),
        step=0.1,
        key="current_setup_solar_kw",
    )
    st.number_input(
        "Current battery size (kWh)",
        min_value=0.0,
        max_value=60.0,
        value=float(st.session_state.get("current_setup_battery_kwh", default_current_setup_battery_kwh)),
        step=0.5,
        key="current_setup_battery_kwh",
    )
    st.number_input(
        "Current battery power (kW)",
        min_value=0.5,
        max_value=30.0,
        value=float(st.session_state.get("current_setup_battery_power_kw", 5.0)),
        step=0.5,
        key="current_setup_battery_power_kw",
    )
    st.checkbox(
        "Current EV enabled",
        value=bool(st.session_state.get("current_setup_ev_enabled", False)),
        key="current_setup_ev_enabled",
    )
    st.number_input(
        "Current EV km/yr",
        min_value=0.0,
        max_value=100000.0,
        value=float(st.session_state.get("current_setup_ev_km_yr", 12000.0)),
        step=500.0,
        key="current_setup_ev_km_yr",
        disabled=not bool(st.session_state.get("current_setup_ev_enabled", False)),
    )

ev_days_map = {"All days": "all", "Weekdays only": "wkday", "Weekends only": "wkend"}
ev_strategy_code = "solar_first_backup" if ev_strategy_label in ("Solar First + Timer Backup", "Solar-first + timer backup") else "timer_grid"
ev_params = EVParams(
    enabled=bool(ev_enabled),
    annual_km=float(ev_annual_km),
    consumption_kwh_per_100km=float(ev_consumption),
    charging_loss_frac=float(ev_loss_pct) / 100.0,
    charger_power_kw=float(ev_charger_kw),
    charge_days=ev_days_map.get(ev_charge_days_label, "all"),
    strategy=ev_strategy_code,
    timer_start=ev_timer_start_t.strftime("%H:%M"),
    timer_end=ev_timer_end_t.strftime("%H:%M"),
    solar_start=ev_solar_start_t.strftime("%H:%M"),
    solar_end=ev_solar_end_t.strftime("%H:%M"),
    backup_start=ev_backup_start_t.strftime("%H:%M"),
    backup_end=ev_backup_end_t.strftime("%H:%M"),
)

df_int, ev_summary = apply_ev_profile_to_intervals(df_int_base, ev_params)
dataset_sig = _dataset_signature(df_int)

solar_profile_for_battery = None
solar_match_pct = 0.0
solar_alignment_quality_pct = 0.0
solar_common_intervals = 0
solar_intervals_total = 0
nem_intervals_total = 0
if uploaded_solar is not None:
    try:
        solar_raw = read_solar_profile_5min(uploaded_solar)
    except Exception as ex:
        solar_raw = pd.DataFrame(columns=["timestamp", "pv_kwh"])
        st.warning(f"Could not parse the solar file: {ex}")

    if solar_raw.empty:
        st.warning("Solar file was loaded but no valid interval rows were detected.")
    else:
        solar_aligned, _solar_match_from_align = align_solar_to_intervals(solar_raw, df_int)
        if solar_aligned.empty:
            st.warning("Solar production file could not be aligned to your NEM12 timestamps.")
        else:
            nem_ts = set(pd.to_datetime(df_int["timestamp"], errors="coerce").dropna().drop_duplicates().tolist())
            solar_ts = set(pd.to_datetime(solar_raw["timestamp"], errors="coerce").dropna().drop_duplicates().tolist())
            solar_common_intervals = int(len(nem_ts & solar_ts))
            solar_intervals_total = int(len(solar_ts))
            nem_intervals_total = int(len(nem_ts))
            solar_alignment_quality_pct = (
                float(solar_common_intervals) / float(solar_intervals_total) * 100.0
                if solar_intervals_total > 0
                else 0.0
            )
            solar_match_pct = (
                float(solar_common_intervals) / float(nem_intervals_total) * 100.0
                if nem_intervals_total > 0
                else 0.0
            )

            solar_profile_for_battery = solar_aligned
            total_pv_kwh = float(solar_aligned["pv_kwh"].sum())

            wide_now = _intervals_wide_from_long(df_int)
            merged = wide_now.merge(solar_aligned, on="timestamp", how="left")
            merged["pv_kwh"] = pd.to_numeric(merged["pv_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
            merged["import_kwh"] = pd.to_numeric(merged["general_kwh"], errors="coerce").fillna(0.0) + pd.to_numeric(merged["controlled_kwh"], errors="coerce").fillna(0.0)
            merged["export_kwh"] = pd.to_numeric(merged["export_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

            total_import_kwh_for_solar = float(merged["import_kwh"].sum())
            total_export_kwh_for_solar = float(merged["export_kwh"].sum())
            total_self_cons_kwh = max(total_pv_kwh - total_export_kwh_for_solar, 0.0)
            total_est_load_kwh = total_import_kwh_for_solar + total_self_cons_kwh
            self_cons_pct = (total_self_cons_kwh / total_pv_kwh * 100.0) if total_pv_kwh > 0 else 0.0
            solar_cov_pct = (total_self_cons_kwh / total_est_load_kwh * 100.0) if total_est_load_kwh > 0 else 0.0
            clipped_excess_export_kwh = float((merged["export_kwh"] - merged["pv_kwh"]).clip(lower=0.0).sum())

            st.info(
                f"Solar production file loaded: **{len(solar_raw):,}** rows. "
                f"**Alignment quality:** {solar_alignment_quality_pct:.1f}% "
                f"({solar_common_intervals:,}/{solar_intervals_total:,} solar intervals align to NEM12). "
                f"**NEM coverage by solar:** {solar_match_pct:.1f}% "
                f"({solar_common_intervals:,}/{nem_intervals_total:,} NEM intervals have solar rows)."
            )
            st.caption(
                "Alignment quality = share of uploaded solar production rows that match a NEM timestamp. "
                "NEM coverage = share of NEM intervals that have a solar row."
            )
            s1, s2, s3 = st.columns(3)
            s1.metric("PV generated", f"{total_pv_kwh:,.1f} kWh")
            s2.metric("Est. self-consumption", f"{self_cons_pct:.1f}%")
            s3.metric("Est. solar coverage of load", f"{solar_cov_pct:.1f}%")
            st.caption(
                "Coverage uses aggregate load: total import + self-consumed PV. "
                "Battery model still preserves your register billing rules."
            )
            st.caption(
                "Est. self-consumption = max(total PV - total export, 0) / total PV. "
                "Est. solar coverage = self-consumed PV / (total import + self-consumed PV)."
            )
            if clipped_excess_export_kwh > 0.01:
                st.caption(
                    f"Reconciliation note: {clipped_excess_export_kwh:,.1f} kWh of intervals had export above PV. "
                    "Self-consumption uses total PV minus total export."
                )

def _default_plan_index(names: list[str], fallback: int = 0) -> int:
    try:
        cur = st.session_state.get("current_retailer", None)
        if cur in names:
            return names.index(cur)
    except Exception:
        pass
    return min(max(int(fallback), 0), max(len(names) - 1, 0))

def _mark_current_plan(name: str) -> str:
    cur = st.session_state.get("current_retailer", None)
    return f"* {name}" if cur and name == cur else name

if ev_summary.get("enabled"):
    ev_mode = "Solar-first + backup" if ev_summary.get("strategy") == "solar_first_backup" else "Timer (grid)"
    st.info(f"EV scenario active: **{ev_mode}** charging profile included in this run.")
    ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
    ev_c1.metric("EV energy requested", f"{float(ev_summary.get('requested_kwh', 0.0)):.1f} kWh")
    ev_c2.metric("Added grid import", f"{float(ev_summary.get('grid_kwh', 0.0)):.1f} kWh")
    ev_c3.metric("Solar diverted to EV", f"{float(ev_summary.get('solar_diverted_kwh', 0.0)):.1f} kWh")
    ev_c4.metric("Target coverage", f"{float(ev_summary.get('coverage_pct', 0.0)):.1f}%")
    if float(ev_summary.get("coverage_pct", 100.0)) < 95.0:
        st.warning(
            "Configured EV windows/charger power could not deliver the full EV target in this dataset period. "
            "Increase charger power or widen charging windows."
        )
    if ev_summary.get("notes"):
        st.caption(str(ev_summary.get("notes")))

st.subheader("Sanity check: totals by register (kWh)")
stats = df_int.groupby("register")["kwh"].agg(["sum", "mean", "max"]).reset_index()
st.dataframe(stats, use_container_width=True)

def _current_plan_name() -> str:
    try:
        return str(st.session_state.get("current_retailer", "") or "")
    except Exception:
        return ""

def _add_current_flag(df: pd.DataFrame, plan_col: str = "Plan") -> pd.DataFrame:
    cur = _current_plan_name()
    if not cur or plan_col not in df.columns:
        return df
    out = df.copy()
    out.insert(0, "Current?", out[plan_col].apply(lambda x: "Yes" if str(x) == cur else ""))
    return out

def _style_current_rows(df: pd.DataFrame, plan_col: str = "Plan"):
    cur = _current_plan_name()
    if not cur or plan_col not in df.columns:
        return df
    def _row_style(row):
        # Highlight the user's current plan row
        is_cur = str(row.get(plan_col, "")) == cur
        return ["background-color: #FFF2CC"] * len(row) if is_cur else [""] * len(row)
    return df.style.apply(_row_style, axis=1)

def _show_dataframe_with_frozen_column(df: pd.DataFrame, freeze_col: str) -> None:
    # Streamlit keeps index visible while horizontally scrolling, so use key column as index.
    if freeze_col in df.columns:
        st.dataframe(df.set_index(freeze_col), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

def _show_comparison_with_frozen_plan(df: pd.DataFrame, plan_col: str = "Plan") -> None:
    _show_dataframe_with_frozen_column(df, freeze_col=plan_col)

# ---------------------------
# Bulk comparison
# ---------------------------

# ---------------------------
# Energy overview (dataset)
# ---------------------------
def _infer_interval_minutes(df: pd.DataFrame) -> float:
    try:
        ts = df["timestamp"].dropna().drop_duplicates().sort_values()
        if len(ts) < 2:
            return 5.0
        diffs = ts.diff().dropna().dt.total_seconds() / 60.0
        m = float(diffs.median())
        return m if m > 0 else 5.0
    except Exception:
        return 5.0

def _max_demand_kw(df: pd.DataFrame, import_registers=("E1","E2")) -> float:
    if df.empty:
        return 0.0
    mins = _infer_interval_minutes(df)
    g = df[df["register"].isin(list(import_registers))].groupby("timestamp")["kwh"].sum()
    if g.empty:
        return 0.0
    return float(g.max() * (60.0 / mins))

days_in_data = int(df_int["timestamp"].dt.date.nunique()) if not df_int.empty else 0
total_import_kwh = float(df_int[df_int["register"].isin(["E1","E2"])]["kwh"].sum()) if not df_int.empty else 0.0
total_export_kwh = float(df_int[df_int["register"].isin(["B1"])]["kwh"].sum()) if not df_int.empty else 0.0
avg_daily_pv_production = None
if isinstance(solar_profile_for_battery, pd.DataFrame) and not solar_profile_for_battery.empty and "pv_kwh" in solar_profile_for_battery.columns:
    total_pv_prod_kwh = float(pd.to_numeric(solar_profile_for_battery["pv_kwh"], errors="coerce").fillna(0.0).sum())
    avg_daily_pv_production = (total_pv_prod_kwh / days_in_data) if days_in_data else 0.0
avg_daily_import = (total_import_kwh / days_in_data) if days_in_data else 0.0
max_demand_30 = compute_max_demand_kw(df_int, window_minutes=30)
max_demand_5 = compute_max_demand_kw(df_int, window_minutes=5)
monthly_max_demand = compute_monthly_max_demand(df_int, window_minutes=30)

st.subheader("Energy overview (dataset)")
if avg_daily_pv_production is not None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Days", f"{days_in_data}")
    c2.metric("Import (E1+E2)", f"{total_import_kwh:,.1f} kWh")
    c3.metric("Export (B1)", f"{total_export_kwh:,.1f} kWh")
    c4.metric("Avg daily PV production", f"{avg_daily_pv_production:,.1f} kWh/day")
    c5.metric("Avg daily import", f"{avg_daily_import:,.1f} kWh/day")
    c6.metric("Max demand (30-min avg)", f"{max_demand_30:,.2f} kW")
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Days", f"{days_in_data}")
    c2.metric("Import (E1+E2)", f"{total_import_kwh:,.1f} kWh")
    c3.metric("Export (B1)", f"{total_export_kwh:,.1f} kWh")
    c4.metric("Avg daily import", f"{avg_daily_import:,.1f} kWh/day")
    c5.metric("Max demand (30-min avg)", f"{max_demand_30:,.2f} kW")
st.caption(f"Max 5-minute interval: {max_demand_5:,.2f} kW")
st.caption(
    "Max demand (30-min avg) is a rolling 30-minute average demand metric; "
    "Max 5-minute interval is the highest single interval demand."
)

st.divider()

# Monthly max demand (30-min average) - matches Energex-style reporting
if 'monthly_max_demand' in locals() and monthly_max_demand is not None and not monthly_max_demand.empty:
    st.subheader("Monthly maximum demand (30-min average)")
    md = monthly_max_demand.copy()
    md["month"] = pd.to_datetime(md["month"]).dt.strftime("%b %Y")
    st.dataframe(md.rename(columns={"month":"Month","max_demand_kw":"Max demand (kW)"}), use_container_width=True)

if show_overview_only:
    st.stop()


st.subheader("Comparison results (GST inclusive)")
st.caption(
    "Effective import (c/kWh) includes fixed charges and nets off FiT credit. "
    "Avg import rate excl supply nets off FiT but excludes fixed charges."
)
st.caption(
    "Where configured, one-time sign-up credit is included once in period totals for this comparison view. "
    "Forecast and optimizer modelling excludes it."
)
if not bool(has_home_battery):
    if bool(include_battery_dependent_in_comparison):
        st.caption("No-battery mode is enabled, but battery-dependent plans are included in this table (override ON).")
    else:
        st.caption("No-battery mode is enabled for this table: battery-dependent plans are excluded.")

comparison_plans = list(plans)
excluded_battery_dependent_plans: list[tuple[Plan, str]] = []
if (not bool(has_home_battery)) and (not bool(include_battery_dependent_in_comparison)):
    comparison_plans = []
    for _plan in plans:
        _reason = _battery_dependent_plan_reason(_plan)
        if _reason:
            excluded_battery_dependent_plans.append((_plan, _reason))
        else:
            comparison_plans.append(_plan)
    if not comparison_plans:
        comparison_plans = list(plans)
        excluded_battery_dependent_plans = []
        st.warning("No plans remained after filtering, so all plans are shown.")

rows = []
base_totals_by_plan = {}
for p in comparison_plans:
    sim = simulate_plan(df_int, p, include_signup_credit=True)
    if ev_summary.get("enabled"):
        sim_base = simulate_plan(df_int_base, p, include_signup_credit=True)
        base_totals_by_plan[p.name] = float(sim_base.get("total_cents", 0.0)) / 100.0

    general_kwh = float(sim["general_kwh"])
    controlled_kwh = float(sim["controlled_kwh"])
    export_kwh = float(sim["export_kwh"])
    import_kwh = float(sim.get("import_kwh", general_kwh + controlled_kwh))

    general_cents = float(sim["general_cents"])
    controlled_cents = float(sim["controlled_cents"])
    import_usage_cents = float(sim.get("import_cents", general_cents + controlled_cents))
    export_credit_cents = float(sim["export_credit_cents"])
    signup_credit_cents = float(sim.get("signup_credit_cents_applied", 0.0))
    supply_cents = float(sim["supply_cents"])
    controlled_supply_cents = float(sim["controlled_supply_cents"])
    total_cents = float(sim["total_cents"])
    days = int(sim["days"])
    single_rate_all_import = bool(sim.get("flat_applies_to_all_import", False))
    night_fit_bonus = _fit_tou_has_night_bonus_rows(
        [b.__dict__ for b in (p.feed_in_tou.bands if p.feed_in_tou else [])],
        float(p.feed_in_flat_cents_per_kwh or 0.0),
    )
    night_bonus_export_kwh = fit_tou_night_bonus_export_kwh(df_int, p) if night_fit_bonus else 0.0

    effective_general_c_per_kwh = (general_cents / general_kwh) if general_kwh > 0 else 0.0
    # Usage-only average import rate (excludes daily supply charges) and nets off FiT credit.
    avg_import_ex_supply_c_per_kwh = ((import_usage_cents - export_credit_cents) / import_kwh) if import_kwh > 0 else 0.0
    # "Effective Rate" includes supply charges (and nets off FiT credit) per total imported kWh.
    effective_rate_c_per_kwh = (total_cents / import_kwh) if import_kwh > 0 else 0.0

    rows.append({
        "Plan": p.name,
        "FiT type": fit_mode_for_plan(p),
        "Days": days,
        "Import kWh (E1+E2)": round(import_kwh, 3),
        "Import Usage ($)": round(import_usage_cents / 100.0, 2),
        "General kWh (E1)": round(general_kwh, 3),
        "General Usage ($)": round(general_cents / 100.0, 2),
        "Effective General Rate (c/kWh)": round(effective_general_c_per_kwh, 2),
        "Avg import rate excl supply (c/kWh)": round(avg_import_ex_supply_c_per_kwh, 2),
        "Effective Rate (c/kWh)": round(effective_rate_c_per_kwh, 2),
        "Controlled kWh (E2)": round(float(sim["controlled_kwh"]), 3),
        "Export kWh (B1)": round(float(sim["export_kwh"]), 3),
        "Supply ($)": round(float(sim["supply_cents"]) / 100.0, 2),
        "Controlled supply ($)": round(float(sim["controlled_supply_cents"]) / 100.0, 2),
        "Controlled usage ($)": round(float(sim["controlled_cents"]) / 100.0, 2),
        "Single-rate applied to all import": "Yes" if single_rate_all_import else "No",
        "Night FiT bonus window": "Yes" if night_fit_bonus else "No",
        "Night bonus export kWh": round(night_bonus_export_kwh, 3),
        "FiT credit ($)": round(float(sim["export_credit_cents"]) / 100.0, 2),
        "Sign-up credit ($)": round(signup_credit_cents / 100.0, 2),
        "Total ($)": round(total_cents / 100.0, 2),
    })

res = pd.DataFrame(rows).sort_values("Total ($)")

night_bonus_mask = (res["Night FiT bonus window"] == "Yes") & (res["Night bonus export kWh"] <= 0.0)
if night_bonus_mask.any():
    plans_requiring_batt = res.loc[night_bonus_mask, "Plan"].astype(str).tolist()
    st.warning(
        "Night FiT bonus benefits generally require battery export. "
        f"No night bonus export was detected for: {', '.join(plans_requiring_batt)}."
    )
if excluded_battery_dependent_plans:
    st.info(
        f"Excluded {len(excluded_battery_dependent_plans)} battery-dependent plan(s) from this retailer comparison "
        "because your setup is set to no home battery."
    )
    with st.expander("Show excluded battery-dependent plans"):
        df_excluded = pd.DataFrame(
            [{"Plan": pp.name, "Reason": rr} for pp, rr in excluded_battery_dependent_plans]
        )
        _show_comparison_with_frozen_plan(df_excluded, plan_col="Plan")
    st.caption(
        "Battery simulation and optimiser tabs still use the full plan library, including battery-dependent plans."
    )

# Highlight current retailer in displayed tables
res_display = res.copy()
if "Plan" in res_display.columns:
    res_display["Plan"] = res_display["Plan"].apply(_mark_current_plan)

# Callout: current plan vs cheapest (over this dataset period)
cur_name = st.session_state.get("current_retailer", None)
if cur_name and (cur_name in list(res["Plan"])):
    cur_total = float(res.loc[res["Plan"] == cur_name, "Total ($)"].iloc[0])
    best_total = float(res["Total ($)"].iloc[0])
    best_plan = str(res["Plan"].iloc[0])
    delta = cur_total - best_total
    st.info(
        f"**Your current retailer:** {cur_name} - **${cur_total:,.2f}**\n\n"
        f"**Cheapest in this run:** {best_plan} - **${best_total:,.2f}**\n\n"
        f"**Difference:** **${delta:,.2f}** over the period shown."
    )
elif cur_name and any(str(pp.name) == str(cur_name) for pp, _ in excluded_battery_dependent_plans):
    st.info(
        f"Your current retailer ({cur_name}) is excluded from this no-battery comparison view. "
        "Enable 'Include battery-dependent plans in Comparison results' in the sidebar to include it."
    )

view = st.radio("View", ["Summary", "Detailed"], horizontal=True)

if view == "Summary":
    summary = res[[
        "Plan",
        "FiT type",
        "Night FiT bonus window",
        "Night bonus export kWh",
        "Total ($)",
        "Supply ($)",
        "Import Usage ($)",
        "Import kWh (E1+E2)",
        "FiT credit ($)",
        "Sign-up credit ($)",
        "Effective Rate (c/kWh)",
        "Days",
    ]].rename(columns={
        "Import Usage ($)": "Usage ($)",
        "Effective Rate (c/kWh)": "Effective import (c/kWh)",
    })
    _show_comparison_with_frozen_plan(summary, plan_col="Plan")
    st.caption("Night bonus export kWh is export volume that falls inside FiT bonus-night windows.")
else:
    _show_comparison_with_frozen_plan(res_display, plan_col="Plan")

if ev_summary.get("enabled") and base_totals_by_plan:
    st.markdown("#### EV impact vs no-EV baseline")
    st.caption("Positive EV impact means the bill increased after adding EV charging assumptions.")
    ev_delta = res[[
        "Plan",
        "Total ($)",
        "Import kWh (E1+E2)",
        "Export kWh (B1)",
    ]].copy()
    ev_delta["Total without EV ($)"] = ev_delta["Plan"].map(base_totals_by_plan)
    ev_delta["EV impact ($)"] = (ev_delta["Total ($)"] - ev_delta["Total without EV ($)"]).round(2)
    ev_delta = ev_delta.rename(
        columns={
            "Total ($)": "Total with EV ($)",
            "Import kWh (E1+E2)": "Import with EV (kWh)",
            "Export kWh (B1)": "Export with EV (kWh)",
        }
    )
    st.dataframe(ev_delta, use_container_width=True)

winner = res.iloc[0]["Plan"]
st.success(f"Cheapest for this dataset: **{winner}**")

if len(res) >= 2:
    diff = float(res.iloc[1]["Total ($)"]) - float(res.iloc[0]["Total ($)"])
    st.info(f"Savings vs next best: **${diff:.2f}** over this period.")

if show_comparison_only:
    st.stop()

# ---------------------------
# Breakdowns
# ---------------------------
st.subheader("Breakdowns")
if show_plan_library_only:
    tab9, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["Plan Library", "Daily totals", "TOU bands (selected plan)", "Invoice-style breakdown", "Plan details", "Forecast", "Risk & volatility", "Scenario & sensitivity", "Battery"]
    )
elif show_battery_only:
    tab8, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab9 = st.tabs(
        ["Battery", "Daily totals", "TOU bands (selected plan)", "Invoice-style breakdown", "Plan details", "Forecast", "Risk & volatility", "Scenario & sensitivity", "Plan Library"]
    )
else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["Daily totals", "TOU bands (selected plan)", "Invoice-style breakdown", "Plan details", "Forecast", "Risk & volatility", "Scenario & sensitivity", "Battery", "Plan Library"]
    )

with tab1:
    chart_df = None
    for p in plans:
        d = daily_totals(df_int, p)[["date", "total_$"]].copy()
        d = d.rename(columns={"total_$": p.name}).set_index("date")
        chart_df = d if chart_df is None else _dedupe_columns(chart_df.merge(d, how="outer", left_index=True, right_index=True, suffixes=("", "_dup")))
    st.line_chart(chart_df)

    st.subheader("Average daily profile (all days)")
    plot_average_24hr_profile_import_export(df_int, solar_profile=solar_profile_for_battery)
    if isinstance(solar_profile_for_battery, pd.DataFrame) and not solar_profile_for_battery.empty:
        st.caption("Production is shown as an additional solar trace so import/export and generation can be compared by hour.")


    win_plan = next(pp for pp in plans if pp.name == winner)
    dwin = daily_totals(df_int, win_plan)
    st.write(f"Daily breakdown (last 14 days) - **{winner}**")
    st.dataframe(dwin.tail(14), use_container_width=True)

with tab2:
    selected_name = st.selectbox("Select a plan to view TOU breakdown", [p.name for p in plans], index=_default_plan_index([p.name for p in plans], 0))
    selected_plan = next(p for p in plans if p.name == selected_name)

    if selected_plan.import_type != "tou":
        st.info("This plan is not TOU (no peak/shoulder/off-peak breakdown).")
    else:
        bd = tou_breakdown_general(df_int, selected_plan.tou).copy()
        if bd.empty:
            st.warning("No general usage data found for E1.")
        else:
            total_general_kwh = float(bd["kwh"].sum()) if "kwh" in bd.columns else 0.0
            bd["$"] = bd["cents"] / 100.0
            bd["% of General"] = bd["kwh"].apply(lambda x: (float(x) / total_general_kwh * 100.0) if total_general_kwh > 0 else 0.0)

            bd["kwh"] = bd["kwh"].map(lambda x: round(float(x), 3))
            bd["rate_c_per_kwh"] = bd["rate_c_per_kwh"].map(lambda x: round(float(x), 3))
            bd["% of General"] = bd["% of General"].map(lambda x: round(float(x), 1))
            bd["$"] = bd["$"].map(lambda x: round(float(x), 2))

            st.dataframe(bd[["band", "kwh", "% of General", "rate_c_per_kwh", "$"]], use_container_width=True)

            # Flat-rate equivalent callout
            total_cents = float(bd["cents"].sum())
            eff_c_per_kwh = (total_cents / total_general_kwh) if total_general_kwh > 0 else 0.0
            st.success(f"Flat-rate equivalent for General usage (E1): **{eff_c_per_kwh:.2f} c/kWh**")

with tab3:
    st.write("Invoice-style line items (like a retailer invoice).")
    # Default this selector to the current retailer (sidebar) when possible.
    # Implementation note: we control the widget value ONLY via st.session_state (no selectbox index),
    # which avoids Streamlit's warning about setting both a default and session_state.
    _cur = st.session_state.get("current_retailer", None)
    _names = [p.name for p in plans]

    if _names:
        # Initialise session state once (or recover if it becomes invalid)
        if ("invoice_plan" not in st.session_state) or (st.session_state.get("invoice_plan") not in _names):
            st.session_state["invoice_plan"] = _cur if _cur in _names else _names[0]

        # Re-sync ONLY when the sidebar current retailer changes (so users can still explore other plans)
        if ("_invoice_last_current" not in st.session_state) or (st.session_state.get("_invoice_last_current") != _cur):
            if _cur in _names:
                st.session_state["invoice_plan"] = _cur
            st.session_state["_invoice_last_current"] = _cur

    selected_plan_name = st.selectbox(
        "Select plan",
        _names,
        key="invoice_plan",
    )

    selected_plan = next(p for p in plans if p.name == selected_plan_name)
    sim = simulate_plan(df_int, selected_plan)
    if bool(sim.get("flat_applies_to_all_import", False)):
        st.info("This flat plan has no controlled-load usage rate, so the single import tariff is applied to all import kWh (E1+E2).")
    if _has_fit_tou(selected_plan):
        st.info("This plan uses FiT TOU: configured FiT windows override the base FiT rate for matching export intervals.")
    li = sim["line_items"].copy()

    li["qty"] = li["qty"].map(lambda x: round(float(x), 3))
    li["rate"] = li["rate"].map(lambda x: round(float(x), 5))
    li["amount"] = li["amount"].map(lambda x: round(float(x), 2))

    st.dataframe(li[["item", "qty", "unit", "rate", "amount"]], use_container_width=True)

    # Optional reconciliation
    with st.expander("Reconciliation mode (optional)"):
        invoice_total = st.number_input("Enter invoice total ($) for this period", value=0.00, step=0.01, format="%.2f")
        if invoice_total > 0:
            model_total = float(sim["total_cents"]) / 100.0
            delta = model_total - float(invoice_total)
            pct = (delta / float(invoice_total) * 100.0) if invoice_total else 0.0
            st.write(f"Model total: **${model_total:.2f}**")
            st.write(f"Difference (model - invoice): **${delta:.2f}** ({pct:+.2f}%)")
            if abs(pct) > 1.0:
                st.warning("Variance > 1%: check bill period alignment, register mapping, or retailer rounding rules.")

with tab4:
    st.write("Plan settings + locked register mapping")
    # keep JSON safe for display
    st.json({
        "Register mapping": {
            "general_import": GENERAL_REGS,
            "controlled_load": CONTROLLED_REGS,
            "export": EXPORT_REGS,
        },
        "Plans": [_plan_to_dict(p) for p in plans],
    })

with tab5:
    st.subheader("Forward Projection")
    st.caption("Forecast totals exclude one-time plan sign-up credits.")

    horizon = st.selectbox(
        "Select forecast horizon",
        ["1 Month (~30 days)", "12 Months (~1 year)"]
    )

    if horizon.startswith("1 Month"):
        future_df = forecast_repeat_last_4_weeks(df_int, days_forward=30)
        method = "Repeat last 4 weeks"
        confidence = "High (short-term replay)"
    else:
        future_df = forecast_replay_last_year(df_int)
        if future_df is None:
            st.warning("Not enough history for full year replay. Falling back to 4-week repeat over 365 days.")
            future_df = forecast_repeat_last_4_weeks(df_int, days_forward=365)
            method = "Repeat last 4 weeks (fallback)"
            confidence = "Medium (limited history)"
        else:
            method = "Replay last year's usage shape"
            confidence = "High (seasonality captured)"

    if future_df.empty:
        st.error("Not enough data to generate forecast.")
    else:
        st.info(f"Projection method: {method}")
        st.info(f"Confidence level: {confidence}")

        forecast_rows = []
        for p in plans:
            simf = simulate_plan(future_df, p, include_signup_credit=False)
            forecast_rows.append({
                "Plan": p.name,
                "Forecast Days": simf["days"],
                "Forecast General kWh": round(simf["general_kwh"], 1),
                "Forecast Controlled kWh": round(simf["controlled_kwh"], 1),
                "Forecast Export kWh": round(simf["export_kwh"], 1),
                "Forecast Total ($)": round(simf["total_cents"] / 100.0, 2),
            })

        forecast_df = pd.DataFrame(forecast_rows).sort_values("Forecast Total ($)")
        st.dataframe(forecast_df, use_container_width=True)

        forecast_winner = forecast_df.iloc[0]["Plan"]
        st.success(f"Cheapest forecasted plan: **{forecast_winner}**")


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If a merge creates *_dup columns, keep non-null values and drop the dup."""
    dup_cols = [c for c in df.columns if str(c).endswith("_dup")]
    for dc in dup_cols:
        base = str(dc)[:-4]
        if base in df.columns:
            df[base] = df[base].combine_first(df[dc])
            df.drop(columns=[dc], inplace=True)
        else:
            df.rename(columns={dc: base}, inplace=True)
    return df


def _sanitize_tou_rows(rows: list[dict]) -> list[dict]:
    """Drop empty/invalid TOU rows and normalize fields so saving never corrupts plans.json."""
    cleaned = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name","")).strip()
        if name == "" or name.lower() in ["none","null","nan"]:
            continue
        days = _norm_days(r.get("days","all"))
        start = _norm_time_str(r.get("start",""))
        end = _norm_time_str(r.get("end",""))
        # validate parseable times
        try:
            if _parse_hhmm(start) is None or _parse_hhmm(end) is None:
                continue
        except Exception:
            continue
        # numeric
        try:
            cents = float(r.get("cents_per_kwh", 0.0) or 0.0)
        except Exception:
            cents = 0.0
        cleaned.append({"name": name.strip().lower(), "cents_per_kwh": cents, "days": days, "start": start, "end": end})
    return cleaned


def _minute_of_day(t: dt.time) -> int:
    return int(t.hour) * 60 + int(t.minute)


def _tou_ranges_for_band(start: dt.time, end: dt.time) -> list[tuple[int, int]]:
    """Return half-open minute ranges [start, end) over a 24h day for one band."""
    s = _minute_of_day(start)
    # In tariff sheets, 23:59 usually means "to end of day".
    e = 1440 if (int(end.hour) == 23 and int(end.minute) == 59) else _minute_of_day(end)
    if s == e:
        return [(0, 1440)]  # full day
    if s < e:
        return [(s, e)]
    # crosses midnight
    return [(s, 1440), (0, e)]


def _mask_to_ranges(mask: list[bool]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        ranges.append((start, i))
    return ranges


def _fmt_minute(m: int) -> str:
    if int(m) >= 1440:
        return "24:00"
    h = int(m) // 60
    mm = int(m) % 60
    return f"{h:02d}:{mm:02d}"


def _fmt_ranges(ranges: list[tuple[int, int]], max_items: int = 8) -> str:
    if not ranges:
        return "None"
    parts = [f"{_fmt_minute(s)}-{_fmt_minute(e)}" for s, e in ranges]
    if len(parts) <= max_items:
        return ", ".join(parts)
    return ", ".join(parts[:max_items]) + f", ... (+{len(parts) - max_items} more)"


def _validate_tou_coverage(rows: list[dict]) -> dict:
    """Validate that each minute of weekday/weekend has exactly one tariff row."""
    buckets = {
        "wkday": [0] * 1440,
        "wkend": [0] * 1440,
    }

    for r in rows:
        days = _norm_days(r.get("days", "all"))
        start = _parse_hhmm(_norm_time_str(r.get("start", "")))
        end = _parse_hhmm(_norm_time_str(r.get("end", "")))
        if start is None or end is None:
            continue

        applies_to = []
        if days == "wkday":
            applies_to = ["wkday"]
        elif days == "wkend":
            applies_to = ["wkend"]
        else:
            applies_to = ["wkday", "wkend"]

        for s, e in _tou_ranges_for_band(start, end):
            for day_key in applies_to:
                for m in range(s, e):
                    buckets[day_key][m] += 1

    out: dict[str, dict] = {}
    overall_ok = True
    for day_key in ("wkday", "wkend"):
        counts = buckets[day_key]
        missing_ranges = _mask_to_ranges([c == 0 for c in counts])
        overlap_ranges = _mask_to_ranges([c > 1 for c in counts])
        day_ok = (len(missing_ranges) == 0) and (len(overlap_ranges) == 0)
        out[day_key] = {
            "ok": day_ok,
            "missing_ranges": missing_ranges,
            "overlap_ranges": overlap_ranges,
            "covered_minutes": int(sum(1 for c in counts if c > 0)),
        }
        overall_ok = overall_ok and day_ok

    return {
        "ok": overall_ok,
        "wkday": out["wkday"],
        "wkend": out["wkend"],
    }


def _unique_name(existing: list[str], base: str) -> str:
    base = (base or "").strip() or "New retailer plan"
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"




with tab6:
    st.subheader("Risk & volatility analysis")
    st.caption("Shows how bills vary month-to-month for each plan (useful for solar households assessing downside risk, not just averages).")

    risk_rows = []
    monthly_wide = None

    for p in plans:
        m = monthly_totals(df_int, p)
        if m.empty:
            continue

        mean = float(m["total_$"].mean())
        std = float(m["total_$"].std(ddof=0)) if len(m) > 1 else 0.0
        worst = float(m["total_$"].max())
        best = float(m["total_$"].min())
        cv = (std / mean * 100.0) if mean > 0 else 0.0

        risk_rows.append({
            "Plan": p.name,
            "Avg monthly ($)": round(mean, 2),
            "Std dev ($)": round(std, 2),
            "Worst month ($)": round(worst, 2),
            "Best month ($)": round(best, 2),
            "Risk index (CV %)": round(cv, 1),
        })

        # Build wide monthly table for charting
        w = m.copy()
        w["month"] = pd.to_datetime(w["month"])
        w = w.rename(columns={"total_$": p.name}).set_index("month")
        monthly_wide = w if monthly_wide is None else _dedupe_columns(
            monthly_wide.merge(w, how="outer", left_index=True, right_index=True, suffixes=("", "_dup"))
        )

    if not risk_rows:
        st.info("Not enough data to compute monthly volatility (need at least one full month of data).")
    else:
        risk_df = pd.DataFrame(risk_rows).sort_values(["Avg monthly ($)", "Risk index (CV %)"]).reset_index(drop=True)

        st.markdown("#### Plan risk summary")
        st.caption("Risk index (CV %) = std dev / average monthly bill. Lower means more stable bills.")
        risk_df_disp = _add_current_flag(risk_df)
        st.dataframe(_style_current_rows(risk_df_disp), use_container_width=True)

        st.markdown("#### Monthly bill history (by plan)")
        if monthly_wide is not None and not monthly_wide.empty:
            st.line_chart(monthly_wide)

        cheapest_avg = risk_df.sort_values("Avg monthly ($)", ascending=True).head(1).iloc[0]
        lowest_vol = risk_df.sort_values("Risk index (CV %)", ascending=True).head(1).iloc[0]

        st.success(
            f"Cheapest on average (monthly): **{cheapest_avg['Plan']}** at **${cheapest_avg['Avg monthly ($)']:.2f}/month**."
        )
        st.info(
            f"Most stable (lowest volatility): **{lowest_vol['Plan']}** with risk index **{lowest_vol['Risk index (CV %)']:.1f}%**."
        )


with tab7:
    st.subheader("Scenario & sensitivity")
    st.caption("Stress-test plan rankings for solar households by adjusting usage/export and escalating tariffs.")

    if df_int is None or df_int.empty:
        st.info("Upload a dataset to enable scenarios.")
    else:
        s1, s2, s3 = st.columns(3)
        with s1:
            usage_multiplier = st.slider(
                "Import usage multiplier (E1+E2)",
                min_value=0.80,
                max_value=1.50,
                value=1.00,
                step=0.05,
                help="Scales imported energy (general + controlled). Example: 1.20 simulates +20% household usage (e.g., new appliance/EV charging).",
            )
        with s2:
            export_multiplier = st.slider(
                "Solar export multiplier (B1)",
                min_value=0.50,
                max_value=1.20,
                value=1.00,
                step=0.05,
                help="Scales exported energy. Example: 0.80 simulates a 20% drop in PV export (e.g., shading, seasonal change, higher self-consumption).",
            )
        with s3:
            tariff_uplift = st.slider(
                "Tariff uplift (%)",
                min_value=0.0,
                max_value=25.0,
                value=0.0,
                step=1.0,
                help="Applies a uniform uplift to selected tariff components below (useful for 'what if prices rise?').",
            ) / 100.0

        a1, a2, a3 = st.columns(3)
        with a1:
            uplift_import_rates = st.checkbox("Apply uplift to import rates", value=True)
        with a2:
            uplift_supply_charges = st.checkbox("Apply uplift to supply/monthly fees", value=False)
        with a3:
            uplift_fit_rates = st.checkbox("Apply uplift to FiT rates", value=False)

        # Scenario-adjust the interval dataset
        df_scn = df_int.copy()
        df_scn.loc[df_scn["register"].isin(["E1", "E2"]), "kwh"] = (
            df_scn.loc[df_scn["register"].isin(["E1", "E2"]), "kwh"] * float(usage_multiplier)
        )
        df_scn.loc[df_scn["register"] == "B1", "kwh"] = (
            df_scn.loc[df_scn["register"] == "B1", "kwh"] * float(export_multiplier)
        )

        def _uplift_plan(p: Plan) -> Plan:
            pp = copy.deepcopy(p)
            u = float(tariff_uplift)

            if u <= 0:
                return pp

            if uplift_supply_charges:
                pp.supply_cents_per_day = float(pp.supply_cents_per_day or 0.0) * (1 + u)
                pp.controlled_supply_cents_per_day = float(pp.controlled_supply_cents_per_day or 0.0) * (1 + u)
                pp.monthly_fee_cents = float(pp.monthly_fee_cents or 0.0) * (1 + u)

            if uplift_import_rates:
                if pp.flat:
                    pp.flat.cents_per_kwh = float(pp.flat.cents_per_kwh or 0.0) * (1 + u)
                if pp.tou and pp.tou.bands:
                    for b in pp.tou.bands:
                        b.cents_per_kwh = float(b.cents_per_kwh or 0.0) * (1 + u)
                pp.controlled_cents_per_kwh = float(pp.controlled_cents_per_kwh or 0.0) * (1 + u)

            if uplift_fit_rates:
                pp.feed_in_flat_cents_per_kwh = float(pp.feed_in_flat_cents_per_kwh or 0.0) * (1 + u)
                if pp.feed_in_tiered:
                    pp.feed_in_tiered.high_rate_cents = float(pp.feed_in_tiered.high_rate_cents or 0.0) * (1 + u)
                    pp.feed_in_tiered.low_rate_cents = float(pp.feed_in_tiered.low_rate_cents or 0.0) * (1 + u)
                if pp.feed_in_tou and pp.feed_in_tou.bands:
                    for b in pp.feed_in_tou.bands:
                        b.cents_per_kwh = float(b.cents_per_kwh or 0.0) * (1 + u)
            return pp

        rows_scn = []
        for p in plans:
            p_scn = _uplift_plan(p)
            sim_scn = simulate_plan(df_scn, p_scn, include_signup_credit=True)

            rows_scn.append({
                "Plan": p.name,
                "Scenario Total ($)": round(float(sim_scn["total_cents"]) / 100.0, 2),
                "Scenario Supply ($)": round(float(sim_scn["supply_cents"]) / 100.0, 2),
                "Scenario Usage ($)": round((float(sim_scn["general_cents"]) + float(sim_scn["controlled_cents"])) / 100.0, 2),
                "Scenario FiT credit ($)": round(float(sim_scn["export_credit_cents"]) / 100.0, 2),
                "Scenario sign-up credit ($)": round(float(sim_scn.get("signup_credit_cents_applied", 0.0)) / 100.0, 2),
                "Scenario Days": int(sim_scn["days"]),
                "Scenario Import (kWh)": round(float(sim_scn["general_kwh"]) + float(sim_scn["controlled_kwh"]), 1),
                "Scenario Export (kWh)": round(float(sim_scn["export_kwh"]), 1),
            })

        df_scn_res = pd.DataFrame(rows_scn).sort_values("Scenario Total ($)").reset_index(drop=True)

        st.subheader("Scenario comparison (GST inclusive)")
        df_scn_res_disp = _add_current_flag(df_scn_res)
        st.dataframe(_style_current_rows(df_scn_res_disp), use_container_width=True)

        if not df_scn_res.empty:
            scn_winner = df_scn_res.iloc[0]["Plan"]
            st.success(f"Cheapest plan under scenario: **{scn_winner}**")

            cur = _current_plan_name()
            if cur and (cur in df_scn_res['Plan'].values):
                cur_total = float(df_scn_res.loc[df_scn_res['Plan']==cur, 'Scenario Total ($)'].iloc[0])
                best_total = float(df_scn_res.loc[df_scn_res['Plan']==scn_winner, 'Scenario Total ($)'].iloc[0])
                st.info(f"**Your current retailer under this scenario:** {cur} - **${cur_total:.2f}**.  ")
                st.info(f"Difference vs scenario cheapest: **${cur_total - best_total:+.2f}** over this period.")

        # Delta vs baseline (same plan names)
        st.subheader("Delta vs baseline")
        base_map = {r["Plan"]: float(r["Total ($)"]) for _, r in res.iterrows()} if "res" in locals() and res is not None and not res.empty else {}
        if base_map:
            df_delta = df_scn_res[["Plan", "Scenario Total ($)"]].copy()
            df_delta["Baseline Total ($)"] = df_delta["Plan"].map(base_map).astype(float)
            df_delta["Delta ($)"] = (df_delta["Scenario Total ($)"] - df_delta["Baseline Total ($)"]).round(2)
            df_delta_disp = _add_current_flag(df_delta.sort_values("Delta ($)"))
            st.dataframe(_style_current_rows(df_delta_disp), use_container_width=True)
        else:
            st.info("Baseline comparison table not available for delta calculation.")


with tab8:
    st.subheader("Battery simulation (Level 1: solar-only charging)")
    st.caption("Simulates a behind-the-meter battery that **only charges from PV surplus (reduces export)** and discharges to offset imports during high-rate periods (no grid charging).")
    st.caption("Battery economics and optimizer metrics exclude one-time plan sign-up credits.")
    if not bool(st.session_state.get("has_home_battery", False)):
        st.info(
            "Battery optimiser still includes battery-only and battery-dependent plans. "
            "The no-battery filter only affects the Comparison results table."
        )
    if isinstance(solar_profile_for_battery, pd.DataFrame) and not solar_profile_for_battery.empty:
        st.info(
            f"Using uploaded solar production file in dispatch calculations "
            f"(alignment quality: {solar_alignment_quality_pct:.1f}%, NEM coverage: {solar_match_pct:.1f}%)."
        )

    if df_int is None or df_int.empty:
        st.info("Upload a dataset to enable battery simulation.")
    else:
        plan_names = [p.name for p in plans_lib] if plans_lib else []
        if not plan_names:
            st.info("No plans loaded.")
        else:
            default_plan_name = plan_names[0]
            sim_plan_name = st.selectbox("Plan to simulate", plan_names, index=_default_plan_index(plan_names, 0))
            plan_obj = next((p for p in plans_lib if p.name == sim_plan_name), plans_lib[0])

            ui_detail = st.radio(
                "Input detail",
                ["Basic", "Advanced"],
                index=0,
                horizontal=True,
                key="battery_ui_detail",
                help="Basic keeps core controls visible. Advanced exposes all tuning inputs.",
            )
            is_advanced_ui = ui_detail == "Advanced"

            st.markdown("**Battery operation settings:**")
            c1, c2 = st.columns(2)
            with c1:
                power_kw = st.number_input(
                    "Battery power (kW)",
                    min_value=0.5,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    key="battery_power_kw",
                    help="Maximum charge/discharge rate of the battery inverter (how fast it can move energy). It is NOT your solar inverter size. Higher kW means the battery can cover bigger short peaks.",
                )
            with c2:
                reserve_pct = st.number_input(
                    "Reserve (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    key="battery_reserve_pct",
                    help="Portion of battery capacity kept as a buffer and not discharged (helps protect battery life and keeps backup headroom).",
                )

            if is_advanced_ui:
                c3, c4 = st.columns(2)
                with c3:
                    roundtrip_eff = st.number_input(
                        "Round-trip efficiency",
                        min_value=0.50,
                        max_value=0.99,
                        value=0.90,
                        step=0.01,
                        format="%.2f",
                        key="battery_roundtrip_eff",
                        help="Overall efficiency from charging then discharging. For example 0.90 means about 10% of energy is lost as heat over a full cycle.",
                    )
                with c4:
                    init_soc_pct = st.number_input(
                        "Initial SoC (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=5.0,
                        key="battery_init_soc_pct",
                        help="Starting state-of-charge at the beginning of the dataset. Over longer datasets this matters less; over short datasets it can affect results.",
                    )
            else:
                with st.expander("Advanced battery controls", expanded=False):
                    roundtrip_eff = st.number_input(
                        "Round-trip efficiency",
                        min_value=0.50,
                        max_value=0.99,
                        value=0.90,
                        step=0.01,
                        format="%.2f",
                        key="battery_roundtrip_eff",
                        help="Overall efficiency from charging then discharging. For example 0.90 means about 10% of energy is lost as heat over a full cycle.",
                    )
                    init_soc_pct = st.number_input(
                        "Initial SoC (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=5.0,
                        key="battery_init_soc_pct",
                        help="Starting state-of-charge at the beginning of the dataset. Over longer datasets this matters less; over short datasets it can affect results.",
                    )
                    st.caption("Tip: Keep defaults unless you are validating specific operating assumptions.")

            st.markdown("**Battery cost & payback:**")
            cost_mode = st.radio(
                "Cost input",
                ["$/kWh (installed)", "Total battery cost ($)"],
                index=0,
                horizontal=True,
                key="battery_cost_mode",
                help="Use $/kWh for scalable sizing assumptions, or Total battery cost for a fixed installed quote.",
            )
            c_cost1, c_cost2 = st.columns(2)
            with c_cost1:
                cost_per_kwh = st.number_input(
                    "Cost per kWh ($/kWh)",
                    min_value=0.0,
                    value=900.0,
                    step=50.0,
                    key="battery_cost_per_kwh",
                    help="Used when Cost input is $/kWh (installed).",
                )
            with c_cost2:
                total_cost_fixed = st.number_input(
                    "Total battery cost ($)",
                    min_value=0.0,
                    value=12000.0,
                    step=500.0,
                    key="battery_total_cost_fixed",
                    help="Used when Cost input is Total battery cost ($).",
                )

            st.markdown("**Financial assumptions:**")

            scen_cols = st.columns([3, 1])
            with scen_cols[0]:
                st.radio(
                    "Assumption scenario",
                    options=["Conservative", "Typical", "High price growth", "Custom"],
                    index=1,
                    horizontal=True,
                    key="scenario_preset",
                    help="Quick presets for price growth and discount rate. Choose Custom to edit the raw values below.",
                )
            with scen_cols[1]:
                with st.popover("View assumptions"):
                    preset = st.session_state.get("scenario_preset", "Typical")
                    if preset in SCENARIO_PRESETS and SCENARIO_PRESETS[preset]:
                        o = SCENARIO_PRESETS[preset]
                        st.write(f"Price growth: **{o['price_growth']*100:.0f}%/yr**")
                        st.write(f"Discount rate: **{o['discount_rate']*100:.0f}%/yr**")
                    else:
                        st.write("Using the Custom values shown below.")

            preset = st.session_state.get("scenario_preset", "Typical")
            custom_scenario = preset == "Custom"
            preset_vals = SCENARIO_PRESETS.get(preset)
            if (not custom_scenario) and isinstance(preset_vals, dict):
                # Keep displayed (disabled) inputs aligned with the selected preset.
                st.session_state["battery_discount_rate_pct"] = float(preset_vals.get("discount_rate", 0.0)) * 100.0
                st.session_state["battery_price_growth_rate_pct"] = float(preset_vals.get("price_growth", 0.0)) * 100.0
            if not custom_scenario:
                if isinstance(preset_vals, dict):
                    st.info(
                        f"Using '{preset}' preset: discount rate {float(preset_vals.get('discount_rate', 0.0)) * 100.0:.1f}% "
                        f"and price growth {float(preset_vals.get('price_growth', 0.0)) * 100.0:.1f}%/yr. "
                        "Switch to Custom to edit those values."
                    )
                else:
                    st.info(
                        f"Using '{preset}' preset values for discount rate and price growth. "
                        "Switch to Custom to edit those values."
                    )

            if is_advanced_ui:
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    battery_life_years = st.number_input(
                        "Battery life (years)",
                        min_value=1,
                        max_value=25,
                        value=10,
                        step=1,
                        key="battery_life_years",
                        help="How many years you expect the battery investment to deliver savings (often aligns with warranty).",
                    )
                with f2:
                    discount_rate_pct = st.number_input(
                        "Discount rate (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=6.0,
                        step=0.5,
                        key="battery_discount_rate_pct",
                        disabled=not custom_scenario,
                        help="Required return used to discount future savings to present value. Higher values reduce NPV.",
                    )
                with f3:
                    degradation_rate_pct = st.number_input(
                        "Annual degradation (%)",
                        min_value=0.0,
                        max_value=10.0,
                        value=2.0,
                        step=0.5,
                        key="battery_degradation_rate_pct",
                        help="Calendar aging decline (%/yr) applied to battery-delivered savings.",
                    )
                with f4:
                    price_growth_rate_pct = st.number_input(
                        "Electricity price growth (%/yr)",
                        min_value=0.0,
                        max_value=15.0,
                        value=0.0,
                        step=0.5,
                        key="battery_price_growth_rate_pct",
                        disabled=not custom_scenario,
                        help="Annual increase in import/export prices used in the long-term cashflow model (0% keeps prices flat).",
                    )
            else:
                battery_life_years = st.number_input(
                    "Battery life (years)",
                    min_value=1,
                    max_value=25,
                    value=10,
                    step=1,
                    key="battery_life_years",
                    help="How many years you expect the battery investment to deliver savings (often aligns with warranty).",
                )
                with st.expander("Advanced financial controls", expanded=False):
                    discount_rate_pct = st.number_input(
                        "Discount rate (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=6.0,
                        step=0.5,
                        key="battery_discount_rate_pct",
                        disabled=not custom_scenario,
                        help="Required return used to discount future savings to present value. Higher values reduce NPV.",
                    )
                    degradation_rate_pct = st.number_input(
                        "Annual degradation (%)",
                        min_value=0.0,
                        max_value=10.0,
                        value=2.0,
                        step=0.5,
                        key="battery_degradation_rate_pct",
                        help="Calendar aging decline (%/yr) applied to battery-delivered savings.",
                    )
                    price_growth_rate_pct = st.number_input(
                        "Electricity price growth (%/yr)",
                        min_value=0.0,
                        max_value=15.0,
                        value=0.0,
                        step=0.5,
                        key="battery_price_growth_rate_pct",
                        disabled=not custom_scenario,
                        help="Annual increase in import/export prices used in the long-term cashflow model (0% keeps prices flat).",
                    )

            aging_model = st.radio(
                "Battery aging model",
                ["Cycle-aware (recommended)", "Calendar-only (legacy)"],
                index=0,
                horizontal=True,
                key="battery_aging_model",
                help="Cycle-aware uses simulated battery cycling to adjust effective life and savings. Calendar-only keeps the previous model.",
            )
            cycle_aware = aging_model.startswith("Cycle-aware")

            raw_presets = battery_assumptions_cfg.get("presets", []) if isinstance(battery_assumptions_cfg, dict) else []
            assumption_presets = [p for p in raw_presets if isinstance(p, dict) and str(p.get("name", "")).strip()]
            if not assumption_presets:
                assumption_presets = _default_battery_assumptions_config().get("presets", [])
            preset_map = {str(p.get("name", "")).strip(): p for p in assumption_presets}
            preset_names = list(preset_map.keys())
            default_assumption_preset = str(
                battery_assumptions_cfg.get("default_preset", preset_names[0] if preset_names else "General Li-ion (typical)")
            ).strip()
            if default_assumption_preset not in preset_names and preset_names:
                default_assumption_preset = preset_names[0]
            preset_options = preset_names + ["Custom"] if preset_names else ["Custom"]
            if "battery_assumption_preset" not in st.session_state:
                st.session_state["battery_assumption_preset"] = default_assumption_preset if preset_names else "Custom"
            if st.session_state.get("battery_assumption_preset") not in preset_options:
                st.session_state["battery_assumption_preset"] = default_assumption_preset if preset_names else "Custom"

            assumption_preset_name = st.selectbox(
                "Battery assumptions preset",
                options=preset_options,
                key="battery_assumption_preset",
                help="Presets are loaded from battery_assumptions.json. Choose Custom to edit values manually.",
            )
            selected_assumption = preset_map.get(str(assumption_preset_name).strip())
            assumption_custom = str(assumption_preset_name) == "Custom"
            if (not assumption_custom) and selected_assumption is not None:
                st.session_state["battery_cycle_life_efc"] = float(selected_assumption.get("cycle_life_efc", 6000.0))
                st.session_state["battery_eol_capacity_pct"] = float(selected_assumption.get("eol_capacity_pct", 80.0))

            a1, a2 = st.columns(2)
            with a1:
                cycle_life_efc = st.number_input(
                    "Cycle life to end-of-life (EFC)",
                    min_value=500.0,
                    max_value=20000.0,
                    value=6000.0,
                    step=250.0,
                    key="battery_cycle_life_efc",
                    disabled=not assumption_custom,
                    help="Equivalent full cycles expected by end-of-life.",
                )
            with a2:
                eol_capacity_pct = st.number_input(
                    "End-of-life usable capacity (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    key="battery_eol_capacity_pct",
                    disabled=not assumption_custom,
                    help="Capacity retained at end-of-life cycle throughput (often around 70-80%).",
                )
            if selected_assumption is not None:
                src = str(selected_assumption.get("source", "")).strip()
                notes = str(selected_assumption.get("notes", "")).strip()
                if src:
                    st.caption(f"Assumption source: {src}")
                if notes:
                    st.caption(notes)
            if cycle_aware:
                st.caption("Cycle-aware mode applies cycle fade and caps modelled battery life when cycle throughput is high.")
            else:
                st.caption("Calendar-only mode ignores cycle-derived wear in NPV/IRR, but cycle stress is still flagged in results.")

            discount_rate = float(discount_rate_pct) / 100.0
            degradation_rate = float(degradation_rate_pct) / 100.0
            price_growth_rate = float(price_growth_rate_pct) / 100.0
            eol_capacity_frac = float(eol_capacity_pct) / 100.0

            # Apply scenario preset overrides (Quick Mode)
            price_growth_rate, discount_rate, _scenario_label = apply_scenario_overrides(price_growth_rate, discount_rate)

            discharge_min = None
            if plan_obj.import_type == "tou" and plan_obj.tou and plan_obj.tou.bands:
                band_rates = [float(b.cents_per_kwh or 0.0) for b in plan_obj.tou.bands]
                rmin, rmax = (min(band_rates), max(band_rates)) if band_rates else (0.0, 100.0)
                default_thr = _default_discharge_threshold_cents(plan_obj)
                if default_thr is None:
                    default_thr = rmax
                rmin_f = float(rmin)
                rmax_f = float(rmax)
                default_thr_f = float(default_thr)

                # Slider requires min < max. If all TOU rates are identical (common for incomplete/new plans),
                # keep a fixed threshold and avoid raising a StreamlitAPIException.
                if rmax_f > rmin_f:
                    thr = min(max(default_thr_f, rmin_f), rmax_f)
                    if is_advanced_ui:
                        discharge_min = st.slider(
                            "Discharge only when import rate >= (c/kWh)",
                            min_value=rmin_f,
                            max_value=rmax_f,
                            value=thr,
                            step=0.1,
                            key="battery_discharge_threshold",
                            help="Higher threshold preserves battery for higher-price periods; lower threshold discharges more often.",
                        )
                    else:
                        with st.expander("Advanced dispatch controls", expanded=False):
                            discharge_min = st.slider(
                                "Discharge only when import rate >= (c/kWh)",
                                min_value=rmin_f,
                                max_value=rmax_f,
                                value=thr,
                                step=0.1,
                                key="battery_discharge_threshold",
                                help="Higher threshold preserves battery for higher-price periods; lower threshold discharges more often.",
                            )
                    st.caption("Default is shoulder (discharge during shoulder and peak). Increase the threshold toward peak to conserve battery cycles.")
                else:
                    discharge_min = rmin_f
                    st.caption(f"All TOU import rates are currently {rmin_f:.1f} c/kWh. Discharge threshold is fixed at this value.")

            st.info("Suggested flow: 1) Quick comparison -> 2) Battery economics (selected retailer) -> 3) Best overall retailer + battery.")
            if "timestamp" in df_int.columns:
                ts_batt = pd.to_datetime(df_int["timestamp"], errors="coerce")
                ts_start = str(ts_batt.min()) if not ts_batt.empty else ""
                ts_end = str(ts_batt.max()) if not ts_batt.empty else ""
            else:
                ts_start = ""
                ts_end = ""

            joint_signature_current = {
                "rows": int(len(df_int)),
                "ts_start": ts_start,
                "ts_end": ts_end,
                "dataset_sig": dict(dataset_sig),
                "ev_enabled": bool(ev_summary.get("enabled", False)),
                "ev_strategy": str(ev_summary.get("strategy", "off")),
                "solar_loaded": bool(isinstance(solar_profile_for_battery, pd.DataFrame) and not solar_profile_for_battery.empty),
                "solar_match_pct": round(float(solar_match_pct or 0.0), 2),
                "solar_alignment_quality_pct": round(float(solar_alignment_quality_pct or 0.0), 2),
                "solar_common_intervals": int(solar_common_intervals or 0),
                "solar_intervals_total": int(solar_intervals_total or 0),
                "nem_intervals_total": int(nem_intervals_total or 0),
                "solar_total_kwh": (
                    round(float(pd.to_numeric(solar_profile_for_battery["pv_kwh"], errors="coerce").fillna(0.0).sum()), 3)
                    if isinstance(solar_profile_for_battery, pd.DataFrame) and not solar_profile_for_battery.empty and "pv_kwh" in solar_profile_for_battery.columns
                    else 0.0
                ),
                "plan_names": tuple(plan_names),
                "plan_to_simulate": str(sim_plan_name),
                "power_kw": round(float(power_kw), 4),
                "roundtrip_eff": round(float(roundtrip_eff), 4),
                "reserve_pct": round(float(reserve_pct), 4),
                "init_soc_pct": round(float(init_soc_pct), 4),
                "discharge_min": (round(float(discharge_min), 4) if discharge_min is not None else None),
                "cost_mode": str(cost_mode),
                "cost_per_kwh": round(float(cost_per_kwh), 4),
                "total_cost_fixed": round(float(total_cost_fixed), 4),
                "scenario": str(st.session_state.get("scenario_preset", "Typical")),
                "battery_aging_model": str(aging_model),
                "battery_assumption_preset": str(assumption_preset_name),
                "battery_cycle_life_efc": round(float(cycle_life_efc), 4),
                "battery_eol_capacity_pct": round(float(eol_capacity_pct), 4),
                "battery_life_years": int(battery_life_years),
                "discount_rate": round(float(discount_rate), 6),
                "degradation_rate": round(float(degradation_rate), 6),
                "price_growth_rate": round(float(price_growth_rate), 6),
                "joint_min_kwh": round(float(st.session_state.get("joint_min_kwh", 5.0)), 4),
                "joint_max_kwh": round(float(st.session_state.get("joint_max_kwh", 20.0)), 4),
                "joint_step_kwh": round(float(st.session_state.get("joint_step_kwh", 3.0)), 4),
                "joint_include_zero": bool(st.session_state.get("joint_include_zero", True)),
                "joint_enable_solar_sweep": bool(st.session_state.get("joint_enable_solar_sweep", False)),
                "joint_current_pv_kw": round(float(st.session_state.get("joint_current_pv_kw", 6.6)), 4),
                "joint_pv_cost_per_kw": round(float(st.session_state.get("joint_pv_cost_per_kw", 1200.0)), 4),
                "joint_pv_min_kw": round(float(st.session_state.get("joint_pv_min_kw", 3.0)), 4),
                "joint_pv_max_kw": round(float(st.session_state.get("joint_pv_max_kw", 13.0)), 4),
                "joint_pv_step_kw": round(float(st.session_state.get("joint_pv_step_kw", 2.0)), 4),
                "joint_pv_include_zero": bool(st.session_state.get("joint_pv_include_zero", False)),
            }

            prev_joint_signature = st.session_state.get("joint_signature")
            if ("joint_best" in st.session_state) and (prev_joint_signature is not None) and (prev_joint_signature != joint_signature_current):
                for k in ("joint_opt_df", "joint_best", "joint_days", "joint_meta", "joint_signature"):
                    st.session_state.pop(k, None)
                st.info("Decision summary was cleared because assumptions changed. Re-run the optimizer in subtab 3.")

            st.markdown("### Decision summary")
            if "joint_best" in st.session_state:
                joint_meta = st.session_state.get("joint_meta") or {}
                run_at = joint_meta.get("run_at")
                run_notes = joint_meta.get("summary")
                if run_at:
                    st.caption(f"Last run: {run_at}")
                if run_notes:
                    st.caption(run_notes)
                best = st.session_state.get("joint_best") or {}
                best_plan = best.get("Plan")
                best_size = best.get("Optimal battery (kWh)")
                best_solar = best.get("Optimal solar size (kW)")
                pv_cost = best.get("PV lifetime cost incl battery ($)")
                pv_save = best.get("PV lifetime saving vs same solar, no battery ($)")
                if pv_save is None:
                    pv_save = best.get("PV lifetime saving vs no battery ($)")
                irr = best.get("IRR (%)")
                batt_npv = best.get("Battery NPV ($)")
                ann = best.get("Annualised savings ($/yr)")
                best_cycle_stress = str(best.get("Cycle stress", "")).strip().lower() == "high"
                best_implied_life = best.get("Implied cycle life (yrs)")
                best_model_life = best.get("Life used in model (yrs)")
                implied_life_disp = "-" if best_implied_life is None or pd.isna(best_implied_life) else f"{float(best_implied_life):.1f}"
                model_life_disp = "-" if best_model_life is None or pd.isna(best_model_life) else f"{float(best_model_life):.1f}"

                # Simple confidence flag based on how much data we have
                days_cov = float(st.session_state.get("joint_days", 0.0) or 0.0)
                if days_cov >= 330:
                    conf = "High"
                elif days_cov >= 60:
                    conf = "Moderate"
                else:
                    conf = "Low"

                # Buy / Don't buy signal (investment attractiveness under your assumptions)
                is_attractive = (batt_npv is not None and float(batt_npv) > 0) and (irr is not None and float(irr) / 100.0 > float(discount_rate or 0.0))

                if best_solar is not None and not pd.isna(best_solar):
                    st.markdown(f"**Recommended setup:** {best_plan} + **{float(best_solar):.1f} kW** solar + **{best_size} kWh** battery")
                else:
                    st.markdown(f"**Recommended setup:** {best_plan} + **{best_size} kWh** battery")
                cols = st.columns(4)
                cost_label = "PV lifetime cost (incl incremental solar+battery)" if (best_solar is not None and not pd.isna(best_solar)) else "PV lifetime cost (incl battery)"
                cols[0].metric(cost_label, f"${pv_cost:,.0f}" if pv_cost is not None else "-")
                cols[1].metric("PV saving vs same solar, no battery", f"${pv_save:,.0f}" if pv_save is not None else "-")
                cols[2].metric("Battery IRR", f"{irr:.1f}%" if irr is not None else "-")
                cols[3].metric("Confidence", conf)

                if is_attractive:
                    st.success("Battery looks **financially attractive** under your assumptions (NPV > 0 and IRR > discount rate).")
                else:
                    st.warning(
                        "Battery doesn't look like a good value with these settings right now. "
                        "This is usually because there isn't enough spare solar export at the times the battery can charge and discharge. "
                        "Check your export amount and timing, then try changing battery cost, battery life, or discount rate assumptions."
                    )
                if ann is not None:
                    st.caption(f"Estimated year-1 savings under the recommended setup: ${float(ann):,.0f}/yr (annualised from your uploaded data).")
                if best_cycle_stress:
                    st.warning(
                        "Cycle stress flag on recommended setup: implied cycle life is materially below assumed life. "
                        f"(implied: {implied_life_disp} yrs, modelled life: {model_life_disp} yrs)"
                    )
            else:
                st.info("Run **Retailer + solar + battery decision optimiser** (subtab 3) to generate a recommendation here.")


            sub_compare, sub_opt, sub_joint, sub_whatif = st.tabs(
                [
                    "Quick comparison",
                    "Battery economics (selected retailer)",
                    "Best overall retailer + solar + battery",
                    "What-if: current vs optimised",
                ]
            )

            with sub_compare:
                st.markdown("**Quick comparison (pick a few battery sizes to compare):**")
                size_options = [0.0, 3.0, 5.0, 7.0, 10.0, 13.5, 15.0, 20.0]
                chosen_sizes = st.multiselect("Sizes", size_options, default=[0.0, 5.0, 10.0, 13.5], key="batt_sizes_compare")
                quick_sizes = sorted(set(float(s) for s in chosen_sizes))
                st.caption(f"Estimated run size: {len(quick_sizes)} simulations.")
                if len(quick_sizes) > 30:
                    st.warning("Large run selected. This may take noticeable time.")

                if st.button("Run comparison", type="primary", key="btn_run_batt_compare"):
                    if not quick_sizes:
                        st.error("Select at least one battery size (kWh) to run the simulation.")
                    else:
                        baseline = simulate_plan(df_int, plan_obj, include_signup_credit=False)
                        base_total = float(baseline.get("total_cents", 0.0))
                        days = float(baseline.get("days", 0.0) or 0.0)
                        wide_base = _intervals_wide_from_long(df_int)

                        rows = []
                        progress = st.progress(0.0)
                        status = st.empty()
                        total_runs = len(quick_sizes)
                        for idx, cap in enumerate(quick_sizes, start=1):
                            status.caption(f"Running size {idx}/{total_runs}: {cap:.1f} kWh")
                            batt = BatteryParams(
                                capacity_kwh=float(cap),
                                power_kw=float(power_kw),
                                roundtrip_eff=float(roundtrip_eff),
                                reserve_frac=float(reserve_pct) / 100.0,
                                initial_soc_frac=float(init_soc_pct) / 100.0,
                                discharge_min_rate_cents=float(discharge_min) if discharge_min is not None else None,
                                charge_from_export_only=True,
                            )
                            sim_b = simulate_plan_with_battery(
                                df_int,
                                plan_obj,
                                batt,
                                baseline=baseline,
                                wide_base=wide_base,
                                solar_profile=solar_profile_for_battery,
                            )

                            tot = float(sim_b.get("total_cents", 0.0))
                            savings = (base_total - tot) / 100.0
                            annual_savings = (savings * (365.0 / days)) if days > 0 else 0.0

                            if cost_mode.startswith("$/kWh"):
                                batt_cost = float(cap) * float(cost_per_kwh)
                            else:
                                batt_cost = float(total_cost_fixed)

                            battery_cycles_equiv = float(sim_b.get("battery_cycles_equiv", 0.0) or 0.0)
                            cycle_metrics = _battery_cycle_metrics(
                                battery_cycles_equiv=battery_cycles_equiv,
                                days=days,
                                assumed_life_years=float(battery_life_years),
                                cycle_life_efc=float(cycle_life_efc),
                            )
                            cf_model = _build_battery_cashflows(
                                annual_savings=float(annual_savings),
                                batt_cost=float(batt_cost),
                                battery_life_years=float(battery_life_years),
                                degradation_rate=float(degradation_rate),
                                price_growth_rate=float(price_growth_rate or 0.0),
                                cycle_aware=bool(cycle_aware),
                                efc_per_year=float(cycle_metrics.get("efc_per_year", 0.0)),
                                cycle_life_efc=float(cycle_life_efc),
                                eol_capacity_frac=float(eol_capacity_frac),
                            )
                            cashflows = list(cf_model.get("cashflows", [-float(batt_cost)]))
                            effective_life_years = float(cf_model.get("effective_life_years", battery_life_years) or 0.0)
                            payback_years = (batt_cost / annual_savings) if annual_savings > 0 else None
                            npv_val = _npv(cashflows, float(discount_rate))
                            irr_val = None if float(cap) <= 0.0 else _irr_bisection(cashflows)

                            discharge_kwh = float(sim_b.get("battery_discharge_kwh", 0.0) or 0.0)
                            value_per_discharged = (savings / discharge_kwh) if discharge_kwh > 0 else None
                            annual_per_installed = (annual_savings / cap) if cap > 0 else None

                            rows.append(
                                {
                                    "Battery (kWh)": cap,
                                    "Total ($)": round(tot / 100.0, 2),
                                    "Savings vs no battery ($)": round(savings, 2),
                                    "Annualised savings ($/yr)": round(annual_savings, 0),
                                    "Battery cost ($)": round(batt_cost, 0),
                                    "Simple payback (yrs)": (round(payback_years, 1) if payback_years is not None else None),
                                    "NPV ($)": round(npv_val, 0),
                                    "IRR (%)": (round(float(irr_val) * 100.0, 1) if irr_val is not None else None),
                                    "Lifetime net ($)": round(sum(cashflows[1:]) - batt_cost, 0),
                                    "Import savings ($)": round(float(sim_b.get("savings_import_cents", 0.0)) / 100.0, 2),
                                    "FiT lost ($)": round(float(sim_b.get("lost_fit_cents", 0.0)) / 100.0, 2),
                                    "Value per kWh discharged ($/kWh)": (round(value_per_discharged, 3) if value_per_discharged is not None else None),
                                    "Annual savings per installed kWh ($/kWh-yr)": (round(annual_per_installed, 3) if annual_per_installed is not None else None),
                                    "Charge from PV (kWh)": round(float(sim_b.get("battery_charge_kwh", 0.0)), 1),
                                    "Discharge to load (kWh)": round(discharge_kwh, 1),
                                    "Cycles equiv": round(battery_cycles_equiv, 2),
                                    "Cycles/yr (EFC)": round(float(cycle_metrics.get("efc_per_year", 0.0)), 1),
                                    "Implied cycle life (yrs)": (
                                        round(float(cycle_metrics["implied_cycle_life_years"]), 1)
                                        if cycle_metrics.get("implied_cycle_life_years") is not None
                                        else None
                                    ),
                                    "Life used in model (yrs)": round(float(effective_life_years), 2),
                                    "Cycle stress": ("High" if cycle_metrics.get("is_cycle_stress") else ""),
                                    "Import reduction (kWh)": round(float(sim_b.get("import_reduction_kwh", 0.0)), 1),
                                    "Export reduction (kWh)": round(float(sim_b.get("export_reduction_kwh", 0.0)), 1),
                                }
                            )
                            progress.progress(float(idx) / float(total_runs))

                        status.empty()
                        progress.empty()
                        st.session_state["batt_quick_df"] = pd.DataFrame(rows).sort_values(["Battery (kWh)"]).reset_index(drop=True)
                        st.session_state["batt_quick_meta"] = {
                            "run_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "summary": (
                                f"Quick comparison; plan={sim_plan_name}; scenario={_scenario_label}; "
                                f"power={float(power_kw):.1f}kW; reserve={float(reserve_pct):.0f}%; "
                                f"eff={float(roundtrip_eff):.2f}; aging={'cycle-aware' if cycle_aware else 'calendar-only'}; "
                                f"assumptions={assumption_preset_name}; "
                                f"simulations={total_runs}"
                            ),
                        }

                df_b = st.session_state.get("batt_quick_df")
                if isinstance(df_b, pd.DataFrame) and not df_b.empty:
                    quick_meta = st.session_state.get("batt_quick_meta") or {}
                    if quick_meta.get("run_at"):
                        st.caption(f"Last run: {quick_meta['run_at']}")
                    if quick_meta.get("summary"):
                        st.caption(quick_meta["summary"])
                    _show_dataframe_with_frozen_column(df_b, freeze_col="Battery (kWh)")
                    st.caption(
                        "Cycles equiv = discharged kWh / battery kWh over the dataset. "
                        "Cycle stress = implied cycle life more than 20% below assumed life."
                    )
                    stress_rows = df_b[df_b["Cycle stress"].astype(str) == "High"] if "Cycle stress" in df_b.columns else pd.DataFrame()
                    if not stress_rows.empty:
                        stress_sizes = ", ".join(f"{float(v):g}" for v in stress_rows["Battery (kWh)"].tolist())
                        st.warning(
                            "Cycle stress detected for battery sizes (kWh): "
                            f"{stress_sizes}. Implied cycle-limited life is materially shorter than assumed life."
                        )

                    best = df_b.sort_values(["NPV ($)", "Battery (kWh)"], ascending=[False, True]).head(1).iloc[0]
                    best_kwh = float(best.get("Battery (kWh)", 0.0) or 0.0)
                    if best_kwh <= 0.0:
                        st.success(
                            f"Recommended outcome (from tested): **No battery (0.0 kWh)** "
                            f"(NPV **${best['NPV ($)']:.0f}**, saves **${best['Savings vs no battery ($)']:.2f}** over the dataset)."
                        )
                    else:
                        st.success(
                            f"Recommended size (from tested): **{best_kwh:g} kWh** "
                            f"(NPV **${best['NPV ($)']:.0f}**, saves **${best['Savings vs no battery ($)']:.2f}** over the dataset)."
                        )
                else:
                    st.info("Run comparison to generate results.")

            with sub_opt:
                st.markdown("**Battery economics (selected retailer):**")
                st.caption("Explore how battery size changes savings, NPV and IRR for the selected retailer.")
                o1, o2, o3 = st.columns(3)
                with o1:
                    opt_min_kwh = st.number_input("Min size (kWh)", min_value=0.0, max_value=60.0, value=5.0, step=0.5, key="opt_min_kwh", help="Smallest battery size included in the sweep.")
                with o2:
                    opt_max_kwh = st.number_input("Max size (kWh)", min_value=0.0, max_value=60.0, value=20.0, step=1.0, key="opt_max_kwh", help="Largest battery size included in the sweep.")
                with o3:
                    opt_step_kwh = st.number_input("Step (kWh)", min_value=0.1, max_value=5.0, value=3.0, step=0.1, key="opt_step_kwh", help="Size increment between tested batteries. Smaller step = finer result but longer run.")
                opt_include_zero = st.checkbox("Include 0 kWh (no battery) in sweep", value=True, key="opt_include_zero")

                mn = float(opt_min_kwh)
                mx = float(opt_max_kwh)
                stp = float(opt_step_kwh)
                opt_sizes = []
                opt_err = None
                if stp <= 0:
                    opt_err = "Step must be > 0."
                elif mx < mn:
                    opt_err = "Max size must be >= Min size."
                else:
                    n_steps = int(round((mx - mn) / stp)) + 1
                    opt_sizes = [round(mn + i * stp, 3) for i in range(max(n_steps, 0))]
                    if opt_include_zero:
                        opt_sizes = sorted(set([0.0] + opt_sizes))
                    else:
                        opt_sizes = [s for s in opt_sizes if s > 0]
                    if not opt_sizes:
                        opt_err = "No battery sizes to simulate with this range."

                st.caption(f"Estimated run size: {len(opt_sizes)} simulations.")
                if len(opt_sizes) > 120:
                    st.warning("Large run selected. This may take noticeable time.")
                if opt_err:
                    st.error(opt_err)

                if st.button("Run battery economics", type="primary", key="btn_opt_batt"):
                    if opt_err:
                        st.error(opt_err)
                    else:
                        baseline = simulate_plan(df_int, plan_obj, include_signup_credit=False)
                        base_total = float(baseline.get("total_cents", 0.0))
                        days = float(baseline.get("days", 0.0) or 0.0)
                        annual_base_bill = (base_total / 100.0) * (365.0 / days) if days > 0 else (base_total / 100.0)
                        wide_base = _intervals_wide_from_long(df_int)
                        rows = []
                        progress = st.progress(0.0)
                        status = st.empty()
                        total_runs = len(opt_sizes)
                        for idx, cap in enumerate(opt_sizes, start=1):
                            status.caption(f"Running size {idx}/{total_runs}: {cap:.1f} kWh")
                            batt = BatteryParams(
                                capacity_kwh=float(cap),
                                power_kw=float(power_kw),
                                roundtrip_eff=float(roundtrip_eff),
                                reserve_frac=float(reserve_pct) / 100.0,
                                initial_soc_frac=float(init_soc_pct) / 100.0,
                                discharge_min_rate_cents=float(discharge_min) if discharge_min is not None else None,
                                charge_from_export_only=True,
                            )
                            sim_b = simulate_plan_with_battery(
                                df_int,
                                plan_obj,
                                batt,
                                baseline=baseline,
                                wide_base=wide_base,
                                solar_profile=solar_profile_for_battery,
                            )
                            tot = float(sim_b.get("total_cents", 0.0))
                            savings = (base_total - tot) / 100.0
                            annual_savings = (savings * (365.0 / days)) if days > 0 else 0.0
                            batt_cost = float(cap) * float(cost_per_kwh) if cost_mode.startswith("$/kWh") else float(total_cost_fixed)
                            battery_cycles_equiv = float(sim_b.get("battery_cycles_equiv", 0.0) or 0.0)
                            cycle_metrics = _battery_cycle_metrics(
                                battery_cycles_equiv=battery_cycles_equiv,
                                days=days,
                                assumed_life_years=float(battery_life_years),
                                cycle_life_efc=float(cycle_life_efc),
                            )
                            cf_model = _build_battery_cashflows(
                                annual_savings=float(annual_savings),
                                batt_cost=float(batt_cost),
                                battery_life_years=float(battery_life_years),
                                degradation_rate=float(degradation_rate),
                                price_growth_rate=float(price_growth_rate or 0.0),
                                cycle_aware=bool(cycle_aware),
                                efc_per_year=float(cycle_metrics.get("efc_per_year", 0.0)),
                                cycle_life_efc=float(cycle_life_efc),
                                eol_capacity_frac=float(eol_capacity_frac),
                            )
                            cashflows = list(cf_model.get("cashflows", [-float(batt_cost)]))
                            effective_life_years = float(cf_model.get("effective_life_years", battery_life_years) or 0.0)
                            npv_val = _npv(cashflows, float(discount_rate))
                            irr_val = None if float(cap) <= 0.0 else _irr_bisection(cashflows)
                            rows.append({
                                "Battery (kWh)": cap,
                                "NPV ($)": float(npv_val),
                                "IRR (%)": (float(irr_val) * 100.0 if irr_val is not None else None),
                                "Annualised savings ($/yr)": float(annual_savings),
                                "Estimated annual bill no battery ($/yr)": float(annual_base_bill),
                                "Estimated annual bill with battery ($/yr)": float(max(0.0, annual_base_bill - annual_savings)),
                                "Battery cost ($)": float(batt_cost),
                                "Cycles equiv": round(battery_cycles_equiv, 2),
                                "Cycles/yr (EFC)": round(float(cycle_metrics.get("efc_per_year", 0.0)), 1),
                                "Implied cycle life (yrs)": (
                                    round(float(cycle_metrics["implied_cycle_life_years"]), 1)
                                    if cycle_metrics.get("implied_cycle_life_years") is not None
                                    else None
                                ),
                                "Life used in model (yrs)": round(float(effective_life_years), 2),
                                "Cycle stress": ("High" if cycle_metrics.get("is_cycle_stress") else ""),
                            })
                            progress.progress(float(idx) / float(total_runs))
                        status.empty(); progress.empty()
                        st.session_state["batt_opt_df"] = pd.DataFrame(rows).sort_values("Battery (kWh)").reset_index(drop=True)
                        st.session_state["batt_opt_meta"] = {
                            "run_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "summary": (
                                f"Battery economics; plan={sim_plan_name}; scenario={_scenario_label}; "
                                f"power={float(power_kw):.1f}kW; reserve={float(reserve_pct):.0f}%; "
                                f"eff={float(roundtrip_eff):.2f}; aging={'cycle-aware' if cycle_aware else 'calendar-only'}; "
                                f"assumptions={assumption_preset_name}; "
                                f"simulations={total_runs}"
                            ),
                        }

                df_opt = st.session_state.get("batt_opt_df")
                if isinstance(df_opt, pd.DataFrame) and not df_opt.empty:
                    opt_meta = st.session_state.get("batt_opt_meta") or {}
                    if opt_meta.get("run_at"):
                        st.caption(f"Last run: {opt_meta['run_at']}")
                    if opt_meta.get("summary"):
                        st.caption(opt_meta["summary"])
                    df_opt_disp = df_opt.round(
                        {
                            "NPV ($)": 0,
                            "IRR (%)": 1,
                            "Annualised savings ($/yr)": 0,
                            "Battery cost ($)": 0,
                            "Cycles equiv": 2,
                            "Cycles/yr (EFC)": 1,
                            "Implied cycle life (yrs)": 1,
                            "Life used in model (yrs)": 2,
                        }
                    )
                    _show_dataframe_with_frozen_column(df_opt_disp, freeze_col="Battery (kWh)")
                    st.caption(
                        "NPV uses the selected discount rate. IRR is the implied return of the battery cashflow. "
                        "Cycle stress = implied cycle life more than 20% below assumed life."
                    )
                    stress_rows_opt = df_opt[df_opt["Cycle stress"].astype(str) == "High"] if "Cycle stress" in df_opt.columns else pd.DataFrame()
                    if not stress_rows_opt.empty:
                        stress_sizes_opt = ", ".join(f"{float(v):g}" for v in stress_rows_opt["Battery (kWh)"].tolist())
                        st.warning(
                            "Cycle stress detected for battery sizes (kWh): "
                            f"{stress_sizes_opt}. Consider lower dispatch intensity or larger battery capacity."
                        )
                    best_row = df_opt.sort_values(["NPV ($)", "Battery (kWh)"], ascending=[False, True]).head(1).iloc[0]
                    irr_disp = "-" if best_row.get("IRR (%)") is None or pd.isna(best_row.get("IRR (%)")) else f"{best_row['IRR (%)']:.1f}%"
                    st.success(f"Optimal size: **{best_row['Battery (kWh)']} kWh** (Max NPV: **${best_row['NPV ($)']:.0f}**, IRR: **{irr_disp}**).")
                    st.line_chart(df_opt[["Battery (kWh)", "NPV ($)"]].copy().set_index("Battery (kWh)"))
                else:
                    st.info("Run battery economics to generate results.")
            with sub_joint:
                st.markdown("**Best overall retailer + solar + battery (decision wizard):**")
                st.caption("Sweeps battery sizes for each retailer and optionally solar sizes, then ranks combinations by lowest PV lifetime cost.")

                solar_available = bool(
                    isinstance(solar_profile_for_battery, pd.DataFrame)
                    and not solar_profile_for_battery.empty
                    and {"timestamp", "pv_kwh"}.issubset(solar_profile_for_battery.columns)
                )
                enable_solar_sweep = st.checkbox(
                    "Sweep solar system size",
                    value=solar_available,
                    key="joint_enable_solar_sweep",
                    disabled=not solar_available,
                    help="Requires uploaded solar production file. When enabled, optimizer searches retailer + solar size + battery size together.",
                )
                if not solar_available:
                    st.info("Upload a solar production file to enable solar size optimization. Running retailer + battery optimization only.")
                    enable_solar_sweep = False

                pv_err = None
                if enable_solar_sweep:
                    sp1, sp2 = st.columns(2)
                    with sp1:
                        current_pv_kw = st.number_input(
                            "Current solar system size (kW)",
                            min_value=0.1,
                            max_value=50.0,
                            value=6.6,
                            step=0.1,
                            key="joint_current_pv_kw",
                            help="Used to scale your uploaded PV profile: scaled PV = uploaded PV x (candidate size / current size).",
                        )
                    with sp2:
                        pv_cost_per_kw = st.number_input(
                            "Solar cost ($/kW installed)",
                            min_value=0.0,
                            value=1200.0,
                            step=50.0,
                            key="joint_pv_cost_per_kw",
                            help="Installed cost per added kW used for incremental solar capex.",
                        )
                    st.caption(
                        "Incremental solar capex is modelled as: max(candidate solar kW - current solar kW, 0) x cost per kW. "
                        "Downsizing assumes no rebate/salvage value."
                    )

                    p1, p2, p3 = st.columns(3)
                    with p1:
                        joint_pv_min_kw = st.number_input("Min solar size (kW)", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="joint_pv_min_kw")
                    with p2:
                        joint_pv_max_kw = st.number_input("Max solar size (kW)", min_value=0.0, max_value=50.0, value=13.0, step=0.5, key="joint_pv_max_kw")
                    with p3:
                        joint_pv_step_kw = st.number_input("Solar step (kW)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="joint_pv_step_kw")
                    joint_pv_include_zero = st.checkbox("Include 0 kW solar in sweep", value=False, key="joint_pv_include_zero")

                    pv_mn = float(joint_pv_min_kw)
                    pv_mx = float(joint_pv_max_kw)
                    pv_stp = float(joint_pv_step_kw)
                    joint_pv_sizes: list[float] = []
                    if pv_stp <= 0:
                        pv_err = "Solar step must be > 0."
                    elif pv_mx < pv_mn:
                        pv_err = "Max solar size must be >= Min solar size."
                    else:
                        pv_steps = int(round((pv_mx - pv_mn) / pv_stp)) + 1
                        joint_pv_sizes = [round(pv_mn + i * pv_stp, 3) for i in range(max(pv_steps, 0))]
                        if joint_pv_include_zero:
                            joint_pv_sizes = sorted(set([0.0] + joint_pv_sizes))
                        if current_pv_kw <= 0:
                            pv_err = "Current solar system size must be > 0 when solar sweep is enabled."
                        elif not joint_pv_sizes:
                            pv_err = "No solar sizes to simulate with this range."
                else:
                    current_pv_kw = float(st.session_state.get("joint_current_pv_kw", 6.6))
                    pv_cost_per_kw = float(st.session_state.get("joint_pv_cost_per_kw", 1200.0))
                    joint_pv_sizes = [current_pv_kw]

                j1, j2, j3 = st.columns(3)
                with j1:
                    joint_min_kwh = st.number_input("Min size (kWh)", min_value=0.0, max_value=60.0, value=5.0, step=0.5, key="joint_min_kwh", help="Smallest battery size tested for every plan.")
                with j2:
                    joint_max_kwh = st.number_input("Max size (kWh)", min_value=0.0, max_value=60.0, value=20.0, step=1.0, key="joint_max_kwh", help="Largest battery size tested for every plan.")
                with j3:
                    joint_step_kwh = st.number_input("Step (kWh)", min_value=0.1, max_value=5.0, value=3.0, step=0.1, key="joint_step_kwh", help="Size increment between tested batteries. Smaller step = finer result but longer run.")
                joint_include_zero = st.checkbox("Include 0 kWh (no battery) in sweep", value=True, key="joint_include_zero")

                mn = float(joint_min_kwh)
                mx = float(joint_max_kwh)
                stp = float(joint_step_kwh)
                joint_sizes = []
                joint_err = None
                if stp <= 0:
                    joint_err = "Step must be > 0."
                elif mx < mn:
                    joint_err = "Max size must be >= Min size."
                else:
                    n_steps = int(round((mx - mn) / stp)) + 1
                    joint_sizes = [round(mn + i * stp, 3) for i in range(max(n_steps, 0))]
                    if joint_include_zero:
                        joint_sizes = sorted(set([0.0] + joint_sizes))
                    else:
                        joint_sizes = [s for s in joint_sizes if s > 0]
                    if not joint_sizes:
                        joint_err = "No battery sizes to simulate with this range."

                run_count = len(joint_sizes) * len(plans_lib) * len(joint_pv_sizes)
                if enable_solar_sweep:
                    st.caption(
                        f"Estimated run size: {run_count} simulations "
                        f"({len(plans_lib)} plans x {len(joint_pv_sizes)} solar sizes x {len(joint_sizes)} battery sizes)."
                    )
                else:
                    st.caption(f"Estimated run size: {run_count} simulations ({len(plans_lib)} plans x {len(joint_sizes)} battery sizes).")
                if run_count > 400:
                    st.warning("Very large run selected. Expect a longer wait.")
                if joint_err:
                    st.error(joint_err)
                if pv_err:
                    st.error(pv_err)

                btn_label = "Find best retailer + solar + battery" if enable_solar_sweep else "Find best retailer + battery"
                if st.button(btn_label, type="primary", key="btn_joint_opt"):
                    if joint_err:
                        st.error(joint_err)
                    elif pv_err:
                        st.error(pv_err)
                    else:
                        results = []
                        total_steps = max(run_count, 1)
                        step_idx = 0
                        overall_progress = st.progress(0.0)
                        plan_progress = st.progress(0.0)
                        status = st.empty()

                        pv_cases = []
                        for pv_kw in joint_pv_sizes:
                            if enable_solar_sweep:
                                pv_scale = (float(pv_kw) / float(current_pv_kw)) if float(current_pv_kw) > 0 else 0.0
                                df_case, solar_case, _pv_meta = apply_pv_scale_to_intervals(
                                    df_int,
                                    solar_profile_for_battery,
                                    pv_scale,
                                )
                                pv_capex = max(float(pv_kw) - float(current_pv_kw), 0.0) * float(pv_cost_per_kw)
                            else:
                                pv_scale = 1.0
                                df_case = df_int
                                solar_case = solar_profile_for_battery
                                pv_capex = 0.0

                            pv_cases.append({
                                "pv_kw": float(pv_kw),
                                "pv_scale": float(pv_scale),
                                "pv_capex": float(pv_capex),
                                "df_int": df_case,
                                "solar_profile": solar_case,
                                "wide_base": _intervals_wide_from_long(df_case),
                            })

                        for plan_idx, p in enumerate(plans_lib, start=1):
                            plan_progress.progress(float(plan_idx - 1) / float(max(len(plans_lib), 1)))
                            best = None

                            for pv_case in pv_cases:
                                pv_kw = float(pv_case["pv_kw"])
                                pv_capex = float(pv_case["pv_capex"])
                                df_case = pv_case["df_int"]
                                baseline = simulate_plan(df_case, p, include_signup_credit=False)
                                base_total = float(baseline.get("total_cents", 0.0))
                                days = float(baseline.get("days", 0.0) or 0.0)
                                annual_base_bill = (base_total / 100.0) * (365.0 / days) if days > 0 else (base_total / 100.0)

                                for size_idx, cap in enumerate(joint_sizes, start=1):
                                    if enable_solar_sweep:
                                        status.caption(
                                            f"Plan {plan_idx}/{len(plans_lib)} ({p.name}) | "
                                            f"solar {pv_kw:.1f} kW | battery {size_idx}/{len(joint_sizes)} ({cap:.1f} kWh)"
                                        )
                                    else:
                                        status.caption(
                                            f"Plan {plan_idx}/{len(plans_lib)} ({p.name}) | "
                                            f"battery {size_idx}/{len(joint_sizes)} ({cap:.1f} kWh)"
                                        )

                                    batt = BatteryParams(
                                        capacity_kwh=float(cap),
                                        power_kw=float(power_kw),
                                        roundtrip_eff=float(roundtrip_eff),
                                        reserve_frac=float(reserve_pct) / 100.0,
                                        initial_soc_frac=float(init_soc_pct) / 100.0,
                                        discharge_min_rate_cents=None,
                                        charge_from_export_only=True,
                                    )
                                    sim_b = simulate_plan_with_battery(
                                        df_case,
                                        p,
                                        batt,
                                        baseline=baseline,
                                        wide_base=pv_case["wide_base"],
                                        solar_profile=pv_case["solar_profile"],
                                    )
                                    tot = float(sim_b.get("total_cents", 0.0))
                                    savings = (base_total - tot) / 100.0
                                    annual_savings = (savings * (365.0 / days)) if days > 0 else 0.0
                                    batt_cost = float(cap) * float(cost_per_kwh) if cost_mode.startswith("$/kWh") else float(total_cost_fixed)

                                    battery_cycles_equiv = float(sim_b.get("battery_cycles_equiv", 0.0) or 0.0)
                                    cycle_metrics = _battery_cycle_metrics(
                                        battery_cycles_equiv=battery_cycles_equiv,
                                        days=days,
                                        assumed_life_years=float(battery_life_years),
                                        cycle_life_efc=float(cycle_life_efc),
                                    )
                                    cf_model = _build_battery_cashflows(
                                        annual_savings=float(annual_savings),
                                        batt_cost=float(batt_cost),
                                        battery_life_years=float(battery_life_years),
                                        degradation_rate=float(degradation_rate),
                                        price_growth_rate=float(price_growth_rate or 0.0),
                                        cycle_aware=bool(cycle_aware),
                                        efc_per_year=float(cycle_metrics.get("efc_per_year", 0.0)),
                                        cycle_life_efc=float(cycle_life_efc),
                                        eol_capacity_frac=float(eol_capacity_frac),
                                    )
                                    savings_by_year = list(cf_model.get("savings_by_year", []))
                                    year_fractions = list(cf_model.get("year_fractions", [1.0] * len(savings_by_year)))
                                    cashflows = list(cf_model.get("cashflows", [-float(batt_cost)]))
                                    effective_life_years = float(cf_model.get("effective_life_years", battery_life_years) or 0.0)

                                    npv_val = _npv(cashflows, float(discount_rate))
                                    irr_val = None if float(cap) <= 0.0 else _irr_bisection(cashflows)
                                    r = float(discount_rate or 0.0)
                                    pv_bills = 0.0
                                    pv_bills_no_batt = 0.0
                                    for yr, (sav, yr_frac) in enumerate(zip(savings_by_year, year_fractions), start=1):
                                        growth = (1.0 + float(price_growth_rate or 0.0)) ** (yr - 1)
                                        annual_base_bill_t = float(annual_base_bill) * float(growth) * float(yr_frac)
                                        disc = ((1.0 + r) ** yr) if (1.0 + r) != 0 else 1.0
                                        pv_bills += max(0.0, annual_base_bill_t - float(sav)) / disc
                                        pv_bills_no_batt += annual_base_bill_t / disc

                                    system_lifetime_cost = float(pv_capex + batt_cost + pv_bills)
                                    system_saving_vs_no_batt = float(pv_bills_no_batt - (batt_cost + pv_bills))

                                    cand = {
                                        "Plan": p.name,
                                        "Optimal solar size (kW)": (pv_kw if enable_solar_sweep else None),
                                        "Optimal battery (kWh)": float(cap),
                                        "Battery NPV ($)": float(npv_val),
                                        "IRR (%)": (float(irr_val) * 100.0 if irr_val is not None else None),
                                        "Annualised savings ($/yr)": float(annual_savings),
                                        "Estimated annual bill with battery ($/yr)": float(max(0.0, annual_base_bill - annual_savings)),
                                        "Incremental solar capex ($)": float(pv_capex),
                                        "Battery cost ($)": float(batt_cost),
                                        "Cycles equiv": round(battery_cycles_equiv, 2),
                                        "Cycles/yr (EFC)": round(float(cycle_metrics.get("efc_per_year", 0.0)), 1),
                                        "Implied cycle life (yrs)": (
                                            round(float(cycle_metrics["implied_cycle_life_years"]), 1)
                                            if cycle_metrics.get("implied_cycle_life_years") is not None
                                            else None
                                        ),
                                        "Life used in model (yrs)": round(float(effective_life_years), 2),
                                        "Cycle stress": ("High" if cycle_metrics.get("is_cycle_stress") else ""),
                                        "PV lifetime cost incl battery ($)": system_lifetime_cost,
                                        "PV lifetime saving vs same solar, no battery ($)": system_saving_vs_no_batt,
                                    }
                                    if best is None:
                                        best = cand
                                    else:
                                        best_cost = float(best.get("PV lifetime cost incl battery ($)", 0.0))
                                        cand_cost = float(cand.get("PV lifetime cost incl battery ($)", 0.0))
                                        best_pv = float(best.get("Optimal solar size (kW)") or 0.0)
                                        cand_pv = float(cand.get("Optimal solar size (kW)") or 0.0)
                                        best_b = float(best.get("Optimal battery (kWh)", 0.0))
                                        cand_b = float(cand.get("Optimal battery (kWh)", 0.0))
                                        if (cand_cost < best_cost) or (
                                            cand_cost == best_cost and (
                                                (cand_pv < best_pv) or (cand_pv == best_pv and cand_b < best_b)
                                            )
                                        ):
                                            best = cand

                                    step_idx += 1
                                    overall_progress.progress(float(step_idx) / float(total_steps))

                            if best is not None:
                                results.append(best)

                        plan_progress.progress(1.0)
                        overall_progress.progress(1.0)
                        status.empty()
                        plan_progress.empty()
                        overall_progress.empty()

                        if not results:
                            st.warning("No results. Check that plans and size ranges are valid.")
                        else:
                            sort_cols = ["PV lifetime cost incl battery ($)"]
                            if enable_solar_sweep:
                                sort_cols.append("Optimal solar size (kW)")
                            sort_cols.append("Optimal battery (kWh)")
                            df_joint_run = pd.DataFrame(results).sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)
                            st.session_state["joint_opt_df"] = df_joint_run
                            st.session_state["joint_best"] = df_joint_run.iloc[0].to_dict()
                            st.session_state["joint_days"] = float(pd.to_datetime(df_int["timestamp"], errors="coerce").dt.date.nunique()) if "timestamp" in df_int.columns else 0.0
                            st.session_state["joint_signature"] = joint_signature_current
                            st.session_state["joint_meta"] = {
                                "run_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "summary": (
                                    f"Joint optimizer; scenario={_scenario_label}; "
                                    f"solar_sweep={'on' if enable_solar_sweep else 'off'}; "
                                    f"power={float(power_kw):.1f}kW; reserve={float(reserve_pct):.0f}%; "
                                    f"eff={float(roundtrip_eff):.2f}; aging={'cycle-aware' if cycle_aware else 'calendar-only'}; "
                                    f"assumptions={assumption_preset_name}; "
                                    f"simulations={run_count}"
                                ),
                            }

                df_joint = st.session_state.get("joint_opt_df")
                if isinstance(df_joint, pd.DataFrame) and not df_joint.empty:
                    joint_meta = st.session_state.get("joint_meta") or {}
                    if joint_meta.get("run_at"):
                        st.caption(f"Last run: {joint_meta['run_at']}")
                    if joint_meta.get("summary"):
                        st.caption(joint_meta["summary"])

                    best = df_joint.iloc[0]
                    payback = None
                    try:
                        sav = float(best.get("Annualised savings ($/yr)", 0.0) or 0.0)
                        bc = float(best.get("Battery cost ($)", 0.0) or 0.0) + float(best.get("Incremental solar capex ($)", 0.0) or 0.0)
                        if sav > 0:
                            payback = bc / sav
                    except Exception:
                        payback = None

                    if "Optimal solar size (kW)" in df_joint.columns and df_joint["Optimal solar size (kW)"].notna().any():
                        s1, s2, s3, s4, s5 = st.columns(5)
                        s1.metric("Best plan", str(best.get("Plan", "-")))
                        s2.metric("Best solar size", f"{float(best.get('Optimal solar size (kW)', 0.0)):.1f} kW")
                        s3.metric("Best battery size", f"{float(best.get('Optimal battery (kWh)', 0.0)):.1f} kWh")
                        s4.metric("Year-1 savings", f"${float(best.get('Annualised savings ($/yr)', 0.0)):.0f}/yr")
                        s5.metric("Payback (simple)", (f"{payback:.1f} yrs" if payback is not None else "-"))
                    else:
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Best plan", str(best.get("Plan", "-")))
                        s2.metric("Best battery size", f"{float(best.get('Optimal battery (kWh)', 0.0)):.1f} kWh")
                        s3.metric("Year-1 savings", f"${float(best.get('Annualised savings ($/yr)', 0.0)):.0f}/yr")
                        s4.metric("Payback (simple)", (f"{payback:.1f} yrs" if payback is not None else "-"))

                    _show_dataframe_with_frozen_column(df_joint, freeze_col="Optimal battery (kWh)")
                    st.caption(
                        "PV lifetime cost incl battery = discounted bill stream + incremental solar capex + battery cost. Lower is better."
                    )
                    st.caption("PV lifetime saving is measured against the same solar size with no battery.")
                    st.caption("Cycle stress = implied cycle life more than 20% below assumed life.")
                    stress_rows_joint = df_joint[df_joint["Cycle stress"].astype(str) == "High"] if "Cycle stress" in df_joint.columns else pd.DataFrame()
                    if not stress_rows_joint.empty:
                        stress_plans = ", ".join(str(v) for v in stress_rows_joint["Plan"].tolist())
                        st.warning(
                            "Cycle stress detected for one or more recommended plan combinations: "
                            f"{stress_plans}. Review cycle assumptions and dispatch intensity."
                        )
                    if "Incremental solar capex ($)" in df_joint.columns:
                        st.caption("Payback uses battery cost + incremental solar capex.")
                    st.caption("Results are cached. Change settings and rerun to refresh this table.")
                else:
                    st.info("Run 'Find best retailer + solar + battery' to generate results.")
            with sub_whatif:
                st.markdown("**What-if: current vs optimised:**")
                st.caption(
                    "Simple mode: compare your current setup against one what-if scenario, "
                    "then benchmark against the optimizer recommendation."
                )

                plan_options_whatif = [p.name for p in plans_lib] if plans_lib else []
                if not plan_options_whatif:
                    st.info("No plans loaded.")
                else:
                    has_solar_profile = bool(
                        isinstance(solar_profile_for_battery, pd.DataFrame)
                        and not solar_profile_for_battery.empty
                        and {"timestamp", "pv_kwh"}.issubset(solar_profile_for_battery.columns)
                    )
                    default_plan_whatif = str(st.session_state.get("current_retailer", sim_plan_name) or sim_plan_name)
                    if default_plan_whatif not in plan_options_whatif:
                        default_plan_whatif = plan_options_whatif[0]
                    default_solar_profile_kw = float(
                        st.session_state.get("current_setup_solar_kw", st.session_state.get("joint_current_pv_kw", 6.6))
                    )
                    default_batt_profile_kwh = float(st.session_state.get("current_setup_battery_kwh", 13.5))
                    default_batt_profile_power_kw = float(st.session_state.get("current_setup_battery_power_kw", 5.0))
                    default_ev_profile_enabled = bool(st.session_state.get("current_setup_ev_enabled", ev_enabled))
                    default_ev_profile_km = float(st.session_state.get("current_setup_ev_km_yr", ev_annual_km))
                    base_solar_kw_for_scaling = float(default_solar_profile_kw)

                    if "whatif_simple_current_plan" not in st.session_state:
                        st.session_state["whatif_simple_current_plan"] = default_plan_whatif
                    if "whatif_simple_current_solar_kw" not in st.session_state:
                        st.session_state["whatif_simple_current_solar_kw"] = float(default_solar_profile_kw)
                    if "whatif_simple_current_batt_kwh" not in st.session_state:
                        st.session_state["whatif_simple_current_batt_kwh"] = float(default_batt_profile_kwh)
                    if "whatif_simple_current_batt_power_kw" not in st.session_state:
                        st.session_state["whatif_simple_current_batt_power_kw"] = float(default_batt_profile_power_kw)
                    if "whatif_simple_current_ev_enabled" not in st.session_state:
                        st.session_state["whatif_simple_current_ev_enabled"] = bool(default_ev_profile_enabled)
                    if "whatif_simple_current_ev_km" not in st.session_state:
                        st.session_state["whatif_simple_current_ev_km"] = float(default_ev_profile_km)

                    sidebar_sig_whatif = (
                        str(default_plan_whatif),
                        round(float(default_solar_profile_kw), 4),
                        round(float(default_batt_profile_kwh), 4),
                        round(float(default_batt_profile_power_kw), 4),
                        bool(default_ev_profile_enabled),
                        round(float(default_ev_profile_km), 2),
                    )
                    last_sidebar_sig_whatif = st.session_state.get("whatif_simple_last_sidebar_sig")
                    if last_sidebar_sig_whatif is None:
                        st.session_state["whatif_simple_current_plan"] = default_plan_whatif
                        st.session_state["whatif_simple_current_solar_kw"] = float(default_solar_profile_kw)
                        st.session_state["whatif_simple_current_batt_kwh"] = float(default_batt_profile_kwh)
                        st.session_state["whatif_simple_current_batt_power_kw"] = float(default_batt_profile_power_kw)
                        st.session_state["whatif_simple_current_ev_enabled"] = bool(default_ev_profile_enabled)
                        st.session_state["whatif_simple_current_ev_km"] = float(default_ev_profile_km)
                        st.session_state["whatif_simple_last_sidebar_sig"] = sidebar_sig_whatif
                    elif tuple(last_sidebar_sig_whatif) != sidebar_sig_whatif:
                        st.session_state["whatif_simple_current_plan"] = default_plan_whatif
                        st.session_state["whatif_simple_current_solar_kw"] = float(default_solar_profile_kw)
                        st.session_state["whatif_simple_current_batt_kwh"] = float(default_batt_profile_kwh)
                        st.session_state["whatif_simple_current_batt_power_kw"] = float(default_batt_profile_power_kw)
                        st.session_state["whatif_simple_current_ev_enabled"] = bool(default_ev_profile_enabled)
                        st.session_state["whatif_simple_current_ev_km"] = float(default_ev_profile_km)
                        st.session_state["whatif_simple_last_sidebar_sig"] = sidebar_sig_whatif

                    if st.button("Use sidebar current setup", key="btn_whatif_sync_sidebar"):
                        st.session_state["whatif_simple_current_plan"] = default_plan_whatif
                        st.session_state["whatif_simple_current_solar_kw"] = float(default_solar_profile_kw)
                        st.session_state["whatif_simple_current_batt_kwh"] = float(default_batt_profile_kwh)
                        st.session_state["whatif_simple_current_batt_power_kw"] = float(default_batt_profile_power_kw)
                        st.session_state["whatif_simple_current_ev_enabled"] = bool(default_ev_profile_enabled)
                        st.session_state["whatif_simple_current_ev_km"] = float(default_ev_profile_km)
                        st.session_state["whatif_simple_last_sidebar_sig"] = sidebar_sig_whatif
                        st.rerun()

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        current_plan_input = st.selectbox(
                            "Current plan",
                            options=plan_options_whatif,
                            index=plan_options_whatif.index(
                                str(st.session_state.get("whatif_simple_current_plan", default_plan_whatif))
                                if str(st.session_state.get("whatif_simple_current_plan", default_plan_whatif)) in plan_options_whatif
                                else default_plan_whatif
                            ),
                            key="whatif_simple_current_plan",
                        )
                    with c2:
                        current_solar_kw_input = st.number_input(
                            "Current solar size (kW)",
                            min_value=0.0,
                            max_value=50.0,
                            value=float(st.session_state.get("whatif_simple_current_solar_kw", default_solar_profile_kw)),
                            step=0.1,
                            key="whatif_simple_current_solar_kw",
                        )
                    with c3:
                        current_batt_kwh_input = st.number_input(
                            "Current battery size (kWh)",
                            min_value=0.0,
                            max_value=60.0,
                            value=float(st.session_state.get("whatif_simple_current_batt_kwh", default_batt_profile_kwh)),
                            step=0.5,
                            key="whatif_simple_current_batt_kwh",
                        )

                    c4, c5, c6 = st.columns(3)
                    with c4:
                        current_batt_power_kw_input = st.number_input(
                            "Current battery power (kW)",
                            min_value=0.5,
                            max_value=30.0,
                            value=float(st.session_state.get("whatif_simple_current_batt_power_kw", default_batt_profile_power_kw)),
                            step=0.5,
                            key="whatif_simple_current_batt_power_kw",
                        )
                    with c5:
                        current_ev_enabled_input = st.checkbox(
                            "EV charging enabled",
                            value=bool(st.session_state.get("whatif_simple_current_ev_enabled", default_ev_profile_enabled)),
                            key="whatif_simple_current_ev_enabled",
                        )
                    with c6:
                        current_ev_km_input = st.number_input(
                            "EV km/yr",
                            min_value=0.0,
                            max_value=100000.0,
                            value=float(st.session_state.get("whatif_simple_current_ev_km", default_ev_profile_km)),
                            step=500.0,
                            key="whatif_simple_current_ev_km",
                            disabled=not bool(current_ev_enabled_input),
                        )

                    whatif_choice = st.radio(
                        "What-if scenario",
                        [
                            "Current setup only",
                            "Higher EV use (+25%)",
                            "Bigger battery (+5 kWh)",
                            "No battery",
                            "Custom",
                        ],
                        horizontal=True,
                        key="whatif_simple_choice",
                    )

                    custom_params = None
                    if whatif_choice == "Custom":
                        with st.expander("Custom what-if inputs", expanded=True):
                            cc1, cc2, cc3 = st.columns(3)
                            with cc1:
                                custom_plan = st.selectbox(
                                    "What-if plan",
                                    options=plan_options_whatif,
                                    index=plan_options_whatif.index(str(current_plan_input)),
                                    key="whatif_simple_custom_plan",
                                )
                            with cc2:
                                custom_solar_kw = st.number_input(
                                    "What-if solar size (kW)",
                                    min_value=0.0,
                                    max_value=50.0,
                                    value=float(current_solar_kw_input),
                                    step=0.1,
                                    key="whatif_simple_custom_solar_kw",
                                )
                            with cc3:
                                custom_batt_kwh = st.number_input(
                                    "What-if battery size (kWh)",
                                    min_value=0.0,
                                    max_value=60.0,
                                    value=float(current_batt_kwh_input),
                                    step=0.5,
                                    key="whatif_simple_custom_batt_kwh",
                                )

                            cc4, cc5, cc6 = st.columns(3)
                            with cc4:
                                custom_batt_power = st.number_input(
                                    "What-if battery power (kW)",
                                    min_value=0.5,
                                    max_value=30.0,
                                    value=float(current_batt_power_kw_input),
                                    step=0.5,
                                    key="whatif_simple_custom_batt_power_kw",
                                )
                            with cc5:
                                custom_ev_enabled = st.checkbox(
                                    "What-if EV enabled",
                                    value=bool(current_ev_enabled_input),
                                    key="whatif_simple_custom_ev_enabled",
                                )
                            with cc6:
                                custom_ev_km = st.number_input(
                                    "What-if EV km/yr",
                                    min_value=0.0,
                                    max_value=100000.0,
                                    value=float(current_ev_km_input),
                                    step=500.0,
                                    key="whatif_simple_custom_ev_km",
                                    disabled=not bool(custom_ev_enabled),
                                )
                        custom_params = {
                            "plan": str(custom_plan),
                            "solar_kw": float(custom_solar_kw),
                            "batt_kwh": float(custom_batt_kwh),
                            "batt_power_kw": float(custom_batt_power),
                            "ev_enabled": bool(custom_ev_enabled),
                            "ev_km": float(custom_ev_km),
                        }

                    def _run_simple_case(
                        case_name: str,
                        plan_name: str,
                        solar_kw: float,
                        batt_kwh: float,
                        batt_power_kw: float,
                        ev_enabled_case: bool,
                        ev_km_case: float,
                    ) -> tuple[dict | None, str | None]:
                        plan_case = next((p for p in plans_lib if p.name == plan_name), None)
                        if plan_case is None:
                            return None, f"{case_name}: plan '{plan_name}' not found."

                        ev_case = EVParams(
                            enabled=bool(ev_enabled_case),
                            annual_km=float(ev_km_case),
                            consumption_kwh_per_100km=float(ev_consumption),
                            charging_loss_frac=float(ev_loss_pct) / 100.0,
                            charger_power_kw=float(ev_charger_kw),
                            charge_days=ev_days_map.get(ev_charge_days_label, "all"),
                            strategy=str(ev_strategy_code),
                            timer_start=ev_timer_start_t.strftime("%H:%M"),
                            timer_end=ev_timer_end_t.strftime("%H:%M"),
                            solar_start=ev_solar_start_t.strftime("%H:%M"),
                            solar_end=ev_solar_end_t.strftime("%H:%M"),
                            backup_start=ev_backup_start_t.strftime("%H:%M"),
                            backup_end=ev_backup_end_t.strftime("%H:%M"),
                        )
                        df_case_ev, _ev_case_summary = apply_ev_profile_to_intervals(df_int_base, ev_case)

                        note = None
                        pv_scale_case = 1.0
                        if has_solar_profile:
                            if float(base_solar_kw_for_scaling) > 0:
                                pv_scale_case = float(solar_kw) / float(base_solar_kw_for_scaling)
                            df_case, solar_case, _ = apply_pv_scale_to_intervals(
                                df_case_ev,
                                solar_profile_for_battery,
                                float(pv_scale_case),
                            )
                        else:
                            df_case = df_case_ev
                            solar_case = None
                            if abs(float(solar_kw) - float(base_solar_kw_for_scaling)) > 0.01:
                                note = f"{case_name}: solar size change ignored (no solar profile uploaded)."

                        baseline_case = simulate_plan(df_case, plan_case, include_signup_credit=False)
                        base_total_case = float(baseline_case.get("total_cents", 0.0))
                        days_case = float(baseline_case.get("days", 0.0) or 0.0)

                        batt_case = BatteryParams(
                            capacity_kwh=float(batt_kwh),
                            power_kw=float(batt_power_kw),
                            roundtrip_eff=float(roundtrip_eff),
                            reserve_frac=float(reserve_pct) / 100.0,
                            initial_soc_frac=float(init_soc_pct) / 100.0,
                            discharge_min_rate_cents=_default_discharge_threshold_cents(plan_case),
                            charge_from_export_only=True,
                        )
                        sim_case = simulate_plan_with_battery(
                            df_case,
                            plan_case,
                            batt_case,
                            baseline=baseline_case,
                            wide_base=_intervals_wide_from_long(df_case),
                            solar_profile=solar_case,
                        )

                        total_case = float(sim_case.get("total_cents", 0.0))
                        savings_case = (base_total_case - total_case) / 100.0
                        annual_savings_case = (savings_case * (365.0 / days_case)) if days_case > 0 else 0.0
                        annual_bill_with_batt_case = ((total_case / 100.0) * (365.0 / days_case)) if days_case > 0 else (total_case / 100.0)

                        if float(batt_kwh) <= 0:
                            batt_cost_case = 0.0
                        elif cost_mode.startswith("$/kWh"):
                            batt_cost_case = float(batt_kwh) * float(cost_per_kwh)
                        else:
                            batt_cost_case = float(total_cost_fixed)

                        cycle_metrics_case = _battery_cycle_metrics(
                            battery_cycles_equiv=float(sim_case.get("battery_cycles_equiv", 0.0) or 0.0),
                            days=days_case,
                            assumed_life_years=float(battery_life_years),
                            cycle_life_efc=float(cycle_life_efc),
                        )
                        cf_model_case = _build_battery_cashflows(
                            annual_savings=float(annual_savings_case),
                            batt_cost=float(batt_cost_case),
                            battery_life_years=float(battery_life_years),
                            degradation_rate=float(degradation_rate),
                            price_growth_rate=float(price_growth_rate or 0.0),
                            cycle_aware=bool(cycle_aware),
                            efc_per_year=float(cycle_metrics_case.get("efc_per_year", 0.0)),
                            cycle_life_efc=float(cycle_life_efc),
                            eol_capacity_frac=float(eol_capacity_frac),
                        )
                        cashflows_case = list(cf_model_case.get("cashflows", [-float(batt_cost_case)]))
                        npv_case = _npv(cashflows_case, float(discount_rate))
                        irr_case = None if float(batt_kwh) <= 0.0 else _irr_bisection(cashflows_case)

                        out = {
                            "Scenario": str(case_name),
                            "Plan": str(plan_name),
                            "Solar size (kW)": round(float(solar_kw), 2),
                            "Battery (kWh)": round(float(batt_kwh), 2),
                            "EV km/yr": round(float(ev_km_case), 0),
                            "Annual bill with battery ($/yr)": round(float(annual_bill_with_batt_case), 0),
                            "Annualised savings ($/yr)": round(float(annual_savings_case), 0),
                            "NPV ($)": round(float(npv_case), 0),
                            "IRR (%)": (round(float(irr_case) * 100.0, 1) if irr_case is not None else None),
                            "Cycles/yr (EFC)": round(float(cycle_metrics_case.get("efc_per_year", 0.0)), 1),
                            "Cycle stress": ("High" if cycle_metrics_case.get("is_cycle_stress") else ""),
                        }
                        return out, note

                    if st.button("Run simple comparison", type="primary", key="btn_run_whatif_simple"):
                        notes: list[str] = []

                        current_row, current_note = _run_simple_case(
                            case_name="Current setup",
                            plan_name=str(current_plan_input),
                            solar_kw=float(current_solar_kw_input),
                            batt_kwh=float(current_batt_kwh_input),
                            batt_power_kw=float(current_batt_power_kw_input),
                            ev_enabled_case=bool(current_ev_enabled_input),
                            ev_km_case=float(current_ev_km_input),
                        )
                        if current_note:
                            notes.append(current_note)

                        rows_simple: list[dict] = []
                        if current_row is not None:
                            rows_simple.append(current_row)

                        if whatif_choice != "Current setup only":
                            if whatif_choice == "Higher EV use (+25%)":
                                whatif_params = {
                                    "plan": str(current_plan_input),
                                    "solar_kw": float(current_solar_kw_input),
                                    "batt_kwh": float(current_batt_kwh_input),
                                    "batt_power_kw": float(current_batt_power_kw_input),
                                    "ev_enabled": bool(current_ev_enabled_input),
                                    "ev_km": float(current_ev_km_input) * 1.25,
                                }
                            elif whatif_choice == "Bigger battery (+5 kWh)":
                                whatif_params = {
                                    "plan": str(current_plan_input),
                                    "solar_kw": float(current_solar_kw_input),
                                    "batt_kwh": float(current_batt_kwh_input) + 5.0,
                                    "batt_power_kw": float(current_batt_power_kw_input),
                                    "ev_enabled": bool(current_ev_enabled_input),
                                    "ev_km": float(current_ev_km_input),
                                }
                            elif whatif_choice == "No battery":
                                whatif_params = {
                                    "plan": str(current_plan_input),
                                    "solar_kw": float(current_solar_kw_input),
                                    "batt_kwh": 0.0,
                                    "batt_power_kw": float(current_batt_power_kw_input),
                                    "ev_enabled": bool(current_ev_enabled_input),
                                    "ev_km": float(current_ev_km_input),
                                }
                            else:
                                whatif_params = custom_params or {
                                    "plan": str(current_plan_input),
                                    "solar_kw": float(current_solar_kw_input),
                                    "batt_kwh": float(current_batt_kwh_input),
                                    "batt_power_kw": float(current_batt_power_kw_input),
                                    "ev_enabled": bool(current_ev_enabled_input),
                                    "ev_km": float(current_ev_km_input),
                                }

                            whatif_row, whatif_note = _run_simple_case(
                                case_name="What-if",
                                plan_name=str(whatif_params["plan"]),
                                solar_kw=float(whatif_params["solar_kw"]),
                                batt_kwh=float(whatif_params["batt_kwh"]),
                                batt_power_kw=float(whatif_params["batt_power_kw"]),
                                ev_enabled_case=bool(whatif_params["ev_enabled"]),
                                ev_km_case=float(whatif_params["ev_km"]),
                            )
                            if whatif_note:
                                notes.append(whatif_note)
                            if whatif_row is not None:
                                rows_simple.append(whatif_row)

                        best_opt = st.session_state.get("joint_best") or {}
                        if isinstance(best_opt, dict) and best_opt:
                            rows_simple.append(
                                {
                                    "Scenario": "Optimiser recommendation",
                                    "Plan": str(best_opt.get("Plan", "-")),
                                    "Solar size (kW)": best_opt.get("Optimal solar size (kW)"),
                                    "Battery (kWh)": best_opt.get("Optimal battery (kWh)"),
                                    "EV km/yr": None,
                                    "Annual bill with battery ($/yr)": best_opt.get("Estimated annual bill with battery ($/yr)"),
                                    "Annualised savings ($/yr)": best_opt.get("Annualised savings ($/yr)"),
                                    "NPV ($)": best_opt.get("Battery NPV ($)"),
                                    "IRR (%)": best_opt.get("IRR (%)"),
                                    "Cycles/yr (EFC)": best_opt.get("Cycles/yr (EFC)"),
                                    "Cycle stress": best_opt.get("Cycle stress", ""),
                                }
                            )

                        if not rows_simple:
                            st.warning("Could not calculate any scenarios. Check your inputs.")
                        else:
                            st.session_state["whatif_simple_df"] = pd.DataFrame(rows_simple)
                            st.session_state["whatif_simple_meta"] = {
                                "run_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "choice": str(whatif_choice),
                            }
                            if notes:
                                st.warning("Notes: " + " | ".join(notes))

                    df_simple = st.session_state.get("whatif_simple_df")
                    if isinstance(df_simple, pd.DataFrame) and not df_simple.empty:
                        meta_simple = st.session_state.get("whatif_simple_meta") or {}
                        if meta_simple.get("run_at"):
                            st.caption(f"Last run: {meta_simple['run_at']}")
                        if meta_simple.get("choice"):
                            st.caption(f"Scenario mode: {meta_simple['choice']}")

                        df_disp = df_simple.copy()
                        cur_match = df_disp[df_disp["Scenario"].astype(str) == "Current setup"]
                        if not cur_match.empty:
                            cur_bill = pd.to_numeric(cur_match["Annual bill with battery ($/yr)"], errors="coerce").iloc[0]
                            cur_npv = pd.to_numeric(cur_match["NPV ($)"], errors="coerce").iloc[0]
                            df_disp["Bill delta vs current ($/yr)"] = (
                                pd.to_numeric(df_disp["Annual bill with battery ($/yr)"], errors="coerce") - cur_bill
                            ).round(0)
                            df_disp["NPV delta vs current ($)"] = (
                                pd.to_numeric(df_disp["NPV ($)"], errors="coerce") - cur_npv
                            ).round(0)

                        _show_dataframe_with_frozen_column(df_disp, freeze_col="Scenario")

                        cur_row = df_disp[df_disp["Scenario"].astype(str) == "Current setup"].head(1)
                        whatif_row = df_disp[df_disp["Scenario"].astype(str) == "What-if"].head(1)
                        opt_row = df_disp[df_disp["Scenario"].astype(str) == "Optimiser recommendation"].head(1)

                        m1, m2, m3 = st.columns(3)
                        if not cur_row.empty:
                            cur_bill_v = pd.to_numeric(cur_row["Annual bill with battery ($/yr)"], errors="coerce").iloc[0]
                            m1.metric("Current annual bill", f"${float(cur_bill_v):,.0f}/yr" if pd.notna(cur_bill_v) else "-")
                        else:
                            m1.metric("Current annual bill", "-")

                        if not whatif_row.empty:
                            w_bill = pd.to_numeric(whatif_row["Annual bill with battery ($/yr)"], errors="coerce").iloc[0]
                            w_delta = pd.to_numeric(whatif_row.get("Bill delta vs current ($/yr)"), errors="coerce").iloc[0]
                            m2.metric(
                                "What-if annual bill",
                                f"${float(w_bill):,.0f}/yr" if pd.notna(w_bill) else "-",
                                (f"${float(w_delta):,.0f}/yr" if pd.notna(w_delta) else None),
                            )
                        else:
                            m2.metric("What-if annual bill", "-")

                        if not opt_row.empty:
                            o_bill = pd.to_numeric(opt_row["Annual bill with battery ($/yr)"], errors="coerce").iloc[0]
                            o_delta = pd.to_numeric(opt_row.get("Bill delta vs current ($/yr)"), errors="coerce").iloc[0]
                            m3.metric(
                                "Optimiser annual bill",
                                f"${float(o_bill):,.0f}/yr" if pd.notna(o_bill) else "-",
                                (f"${float(o_delta):,.0f}/yr" if pd.notna(o_delta) else None),
                            )
                        else:
                            m3.metric("Optimiser annual bill", "Run subtab 3")
                    else:
                        st.info("Set your current details, choose one what-if scenario, then click 'Run simple comparison'.")
with tab9:
    st.subheader("Plan Library")
    st.caption("Flow: 1) pick a plan, 2) edit it, 3) save. Add/duplicate/delete save immediately.")

    plans_lib = st.session_state["plans_lib"]
    if not plans_lib:
        st.warning("No plans found in library.")
        st.stop()

    names = [p.name for p in plans_lib]
    cur_name = str(st.session_state.get("current_retailer", "") or "")

    # Backward compatibility for older state and actions that force selection.
    forced_name = st.session_state.pop("force_select_plan", None)
    forced_idx = st.session_state.pop("force_select_plan_idx", None)
    if isinstance(forced_name, str) and forced_name in names:
        st.session_state["plan_library_selected_idx"] = names.index(forced_name)
    if isinstance(forced_idx, int) and 0 <= forced_idx < len(plans_lib):
        st.session_state["plan_library_selected_idx"] = forced_idx

    try:
        selected_idx_state = int(st.session_state.get("plan_library_selected_idx", -1))
    except Exception:
        selected_idx_state = -1
    if selected_idx_state < 0 or selected_idx_state >= len(plans_lib):
        selected_idx_state = names.index(cur_name) if cur_name in names else 0
    st.session_state["plan_library_selected_idx"] = selected_idx_state

    selected_idx = int(
        st.selectbox(
            "Step 1: Select a plan to edit",
            options=list(range(len(plans_lib))),
            key="plan_library_selected_idx",
            format_func=lambda i: f"{plans_lib[i].name} (current)" if plans_lib[i].name == cur_name else plans_lib[i].name,
            help="This selector stays pinned while editing and is no longer reset by the sidebar dropdown.",
        )
    )
    p = plans_lib[selected_idx]

    hdr_c1, hdr_c2 = st.columns([2, 1])
    with hdr_c1:
        st.caption(f"Editing: {p.name}")
    with hdr_c2:
        if st.button("Set selected as current", use_container_width=True):
            st.session_state["_pending_current_retailer"] = p.name
            st.session_state["_pending_invoice_plan"] = p.name
            st.rerun()

    st.divider()

    st.markdown("### Step 2: Add a new plan")
    with st.form("add_plan_form", clear_on_submit=False):
        add_c1, add_c2 = st.columns([2, 1])
        with add_c1:
            new_plan_name = st.text_input("Name", value="New retailer plan", key="new_plan_name_inline")
        with add_c2:
            new_plan_type = st.selectbox("Type", ["flat", "tou"], index=0, key="new_plan_type_inline")
        add_submit = st.form_submit_button("Add plan", type="primary", use_container_width=True)

    if add_submit:
        np_name = _unique_name([x.name for x in plans_lib], (new_plan_name or "").strip() or "New retailer plan")
        np = Plan(
            name=np_name,
            supply_cents_per_day=0.0,
            import_type=new_plan_type,
            flat=FlatTariff(0.0) if new_plan_type == "flat" else None,
            tou=TouTariff(bands=[]) if new_plan_type == "tou" else None,
            controlled_supply_cents_per_day=0.0,
            controlled_cents_per_kwh=0.0,
            feed_in_flat_cents_per_kwh=0.0,
            feed_in_tiered=None,
            monthly_fee_cents=0.0,
            signup_credit_cents=0.0,
        )
        plans_lib.append(np)
        save_plans(plans_lib)
        st.session_state["force_select_plan_idx"] = len(plans_lib) - 1
        st.success("Added and saved.")
        st.rerun()

    st.divider()

    st.markdown("### Step 3: Edit selected plan")
    st.caption("Duplicate and delete apply to the selected plan and save immediately.")
    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        dup_clicked = st.button("Duplicate selected", use_container_width=True)
    with a2:
        del_clicked = st.button("Delete selected", use_container_width=True)
    with a3:
        if st.button("Repair plans.json", use_container_width=True):
            fixed_plans = load_plans([ORIGIN, ALINTA])
            save_plans(fixed_plans)
            st.session_state["plans_lib"] = fixed_plans
            st.session_state["force_select_plan_idx"] = 0
            st.success("Repaired and re-saved plans.json (invalid rows removed, names de-duplicated).")
            st.rerun()

    if dup_clicked:
        copy_plan = _dict_to_plan(_plan_to_dict(p))
        copy_plan.name = _unique_name([x.name for x in plans_lib], f"{p.name} (copy)")
        plans_lib.append(copy_plan)
        save_plans(plans_lib)
        st.session_state["force_select_plan_idx"] = len(plans_lib) - 1
        st.success("Duplicated and saved.")
        st.rerun()

    if del_clicked:
        if len(plans_lib) <= 1:
            st.warning("At least one plan must remain in the library.")
        else:
            del_name = p.name
            was_current = st.session_state.get("current_retailer") == del_name
            was_invoice = st.session_state.get("invoice_plan") == del_name
            plans_lib.pop(selected_idx)
            remaining_names = [x.name for x in plans_lib]
            fallback = remaining_names[0] if remaining_names else ""
            if was_current:
                st.session_state["_pending_current_retailer"] = fallback
            if was_invoice:
                cur_now = str(st.session_state.get("current_retailer", "") or "")
                inv_fallback = cur_now if cur_now in remaining_names else fallback
                st.session_state["_pending_invoice_plan"] = inv_fallback
            save_plans(plans_lib)
            st.session_state["force_select_plan_idx"] = min(selected_idx, len(plans_lib) - 1)
            st.warning("Deleted and saved.")
            st.rerun()

    plan_slug = re.sub(r"[^a-z0-9]+", "_", p.name.lower()).strip("_") or "plan"
    edit_key_prefix = f"plan_{selected_idx}_{plan_slug}"

    def _ek(suffix: str) -> str:
        return f"{edit_key_prefix}_{suffix}"

    st.markdown(f"#### Editing fields for: **{p.name}**")
    new_name = st.text_input(
        "Plan name",
        value=p.name,
        key=_ek("name"),
        help="If this name already exists, a numeric suffix is added on save.",
    )

    import_type = st.selectbox(
        "Import type",
        ["flat", "tou"],
        index=0 if p.import_type == "flat" else 1,
        key=_ek("import_type"),
        help="How general import is billed: one flat rate or time-of-use bands.",
    )

    supply_cpd = st.number_input(
        "Supply charge (c/day)",
        value=float(p.supply_cents_per_day),
        step=0.001,
        format="%.3f",
        key=_ek("supply_cpd"),
        help="Daily fixed charge on the bill, independent of usage.",
    )

    monthly_fee = st.number_input(
        "Monthly fee (c/month)",
        value=float(p.monthly_fee_cents),
        step=0.01,
        format="%.2f",
        key=_ek("monthly_fee"),
        help="Monthly fixed fee (if any) shown on the plan facts/bill.",
    )

    signup_credit_dollars = st.number_input(
        "One-time sign-up credit ($)",
        value=float(p.signup_credit_cents or 0.0) / 100.0,
        step=1.0,
        format="%.2f",
        key=_ek("signup_credit"),
        help="One-off credit applied for plan-switch comparison only. Excluded from forecast and optimizer modelling.",
    )

    st.markdown("#### Controlled load")
    ctl_supply = st.number_input(
        "Controlled supply (c/day)",
        value=float(p.controlled_supply_cents_per_day),
        step=0.001,
        format="%.3f",
        key=_ek("ctl_supply"),
        help="Daily fixed charge for controlled-load meter/circuit (if applicable).",
    )
    ctl_rate = st.number_input(
        "Controlled usage (c/kWh)",
        value=float(p.controlled_cents_per_kwh),
        step=0.001,
        format="%.3f",
        key=_ek("ctl_rate"),
        help="Usage rate for controlled-load energy (typically hot water/managed load).",
    )

    st.markdown("#### Feed-in Tariff (FiT)")
    fit_mode_default = "tou" if _has_fit_tou(p) else ("tiered" if p.feed_in_tiered else "flat")
    fit_mode_options = ["flat", "tiered", "tou"]
    fit_mode = st.selectbox(
        "FiT type",
        fit_mode_options,
        index=fit_mode_options.index(fit_mode_default),
        key=_ek("fit_mode"),
        help="Export credit structure: single rate, daily tiered cap, or time-of-use FiT windows.",
    )

    fit_flat = float(p.feed_in_flat_cents_per_kwh or 0.0)
    fit_high = float(p.feed_in_tiered.high_rate_cents) if p.feed_in_tiered else 10.0
    fit_kwh_day = float(p.feed_in_tiered.high_kwh_per_day) if p.feed_in_tiered else 10.0
    fit_low = float(p.feed_in_tiered.low_rate_cents) if p.feed_in_tiered else 3.0
    fit_tou_rows_clean: list[dict] = []
    fit_tou_has_overlap = False

    if fit_mode == "flat":
        fit_flat = st.number_input(
            "FiT flat rate (c/kWh)",
            value=float(fit_flat),
            step=0.001,
            format="%.3f",
            key=_ek("fit_flat"),
            help="Single export credit rate applied to all exported kWh.",
        )
    elif fit_mode == "tiered":
        fit_high = st.number_input(
            "FiT high rate (c/kWh)",
            value=float(fit_high),
            step=0.001,
            format="%.3f",
            key=_ek("fit_high"),
            help="Higher export credit rate for kWh within the daily FiT cap.",
        )
        fit_kwh_day = st.number_input(
            "FiT cap (kWh/day)",
            value=float(fit_kwh_day),
            step=0.01,
            format="%.2f",
            key=_ek("fit_cap"),
            help="Daily export amount eligible for the high FiT rate before low rate applies.",
        )
        fit_low = st.number_input(
            "FiT low rate (c/kWh)",
            value=float(fit_low),
            step=0.001,
            format="%.3f",
            key=_ek("fit_low"),
            help="Export credit rate applied after the daily FiT cap is exceeded.",
        )
    else:
        fit_flat = st.number_input(
            "Base FiT rate outside bonus windows (c/kWh)",
            value=float(fit_flat),
            step=0.001,
            format="%.3f",
            key=_ek("fit_base_flat"),
            help="If an export timestamp does not match a FiT TOU row, this base rate is used.",
        )
        fit_bands = [b.__dict__ for b in (p.feed_in_tou.bands if p.feed_in_tou else [])]
        if not fit_bands:
            fit_bands = [
                {"name": "bonus night", "cents_per_kwh": 0.0, "days": "all", "start": "00:00", "end": "06:00"},
            ]
        st.caption("FiT TOU bands: rates here override the base FiT during matching windows.")
        fit_tou_df = st.data_editor(
            pd.DataFrame(fit_bands),
            num_rows="dynamic",
            use_container_width=True,
            key=_ek("fit_tou_editor"),
        )
        fit_tou_rows_clean = _sanitize_tou_rows(fit_tou_df.to_dict("records"))
        fit_tou_validation = _validate_tou_coverage(fit_tou_rows_clean)
        wkday_overlap = fit_tou_validation.get("wkday", {}).get("overlap_ranges", [])
        wkend_overlap = fit_tou_validation.get("wkend", {}).get("overlap_ranges", [])
        fit_tou_has_overlap = bool(wkday_overlap or wkend_overlap)
        if fit_tou_has_overlap:
            st.error("FiT TOU overlaps detected. Remove overlaps before saving.")
            if wkday_overlap:
                st.warning(f"FiT weekday overlaps: {_fmt_ranges(wkday_overlap)}")
            if wkend_overlap:
                st.warning(f"FiT weekend overlaps: {_fmt_ranges(wkend_overlap)}")
        else:
            st.success("FiT TOU validation passed: no overlapping windows.")
        wkday_missing = fit_tou_validation.get("wkday", {}).get("missing_ranges", [])
        wkend_missing = fit_tou_validation.get("wkend", {}).get("missing_ranges", [])
        if wkday_missing or wkend_missing:
            st.caption("FiT TOU does not cover all times. Uncovered timestamps use the base FiT rate above.")
        if _fit_tou_has_night_bonus_rows(fit_tou_rows_clean, fit_flat):
            st.warning("Night FiT bonus windows usually require a battery that can export energy to the grid.")

    st.markdown("#### General usage rates")
    flat_rate = float(p.flat.cents_per_kwh) if p.flat else 0.0
    if import_type == "flat":
        flat_rate = st.number_input(
            "Flat general usage (c/kWh)",
            value=float(flat_rate),
            step=0.001,
            format="%.3f",
            key=_ek("flat_rate"),
        )
        if float(flat_rate) == 0.0:
            st.warning("Flat usage rate is 0.00 c/kWh. This plan will show $0 usage until you set a rate.")

    tou_df = None
    tou_rows_clean: list[dict] = []
    tou_validation = None
    if import_type == "tou":
        bands = [b.__dict__ for b in (p.tou.bands if p.tou else [])]
        if not bands:
            bands = [
                {"name": "off peak", "cents_per_kwh": 0.0, "days": "all", "start": "00:00", "end": "07:00"},
                {"name": "shoulder", "cents_per_kwh": 0.0, "days": "all", "start": "07:00", "end": "16:00"},
                {"name": "peak", "cents_per_kwh": 0.0, "days": "all", "start": "16:00", "end": "20:00"},
                {"name": "shoulder", "cents_per_kwh": 0.0, "days": "all", "start": "20:00", "end": "22:00"},
                {"name": "off peak", "cents_per_kwh": 0.0, "days": "all", "start": "22:00", "end": "00:00"},
            ]
        st.caption("TOU bands: add/remove rows. 'days' must be wkday/wkend/all. Times are HH:MM.")
        tou_df = st.data_editor(
            pd.DataFrame(bands),
            num_rows="dynamic",
            use_container_width=True,
            key=_ek("tou_editor"),
        )
        tou_rows_clean = _sanitize_tou_rows(tou_df.to_dict("records"))
        tou_validation = _validate_tou_coverage(tou_rows_clean)

        if tou_validation.get("ok", False):
            st.success("TOU validation passed: weekday and weekend have full 24h coverage with no overlaps.")
        else:
            st.error("TOU validation failed. Fix gaps/overlaps before saving.")

        wkday_info = tou_validation.get("wkday", {})
        wkend_info = tou_validation.get("wkend", {})

        if wkday_info.get("missing_ranges"):
            st.warning(f"Weekday gaps: {_fmt_ranges(wkday_info['missing_ranges'])}")
        if wkday_info.get("overlap_ranges"):
            st.warning(f"Weekday overlaps: {_fmt_ranges(wkday_info['overlap_ranges'])}")
        if wkend_info.get("missing_ranges"):
            st.warning(f"Weekend gaps: {_fmt_ranges(wkend_info['missing_ranges'])}")
        if wkend_info.get("overlap_ranges"):
            st.warning(f"Weekend overlaps: {_fmt_ranges(wkend_info['overlap_ranges'])}")

    tou_can_save = (import_type != "tou") or bool(tou_validation and tou_validation.get("ok", False))
    if import_type == "tou" and not tou_can_save:
        st.info("Save is disabled until TOU covers all weekday/weekend timeslots with exactly one tariff.")
    fit_tou_can_save = (fit_mode != "tou") or (not fit_tou_has_overlap)
    if fit_mode == "tou" and not fit_tou_can_save:
        st.info("Save is disabled until FiT TOU has no overlapping timeslots.")

    requested_name = (new_name or "").strip() or p.name
    other_names = [x.name for i, x in enumerate(plans_lib) if i != selected_idx]
    final_name = _unique_name(other_names, requested_name)

    preview_plan = Plan(
        name=final_name,
        supply_cents_per_day=float(supply_cpd),
        import_type=import_type,
        flat=FlatTariff(float(flat_rate)) if import_type == "flat" else None,
        tou=TouTariff(bands=[TouBand(**row) for row in tou_rows_clean]) if import_type == "tou" else None,
        controlled_supply_cents_per_day=float(ctl_supply),
        controlled_cents_per_kwh=float(ctl_rate),
        feed_in_flat_cents_per_kwh=float(fit_flat if fit_mode in ("flat", "tou") else 0.0),
        feed_in_tiered=TieredFiT(float(fit_high), float(fit_kwh_day), float(fit_low)) if fit_mode == "tiered" else None,
        feed_in_tou=TouTariff(bands=[TouBand(**row) for row in fit_tou_rows_clean]) if fit_mode == "tou" else None,
        monthly_fee_cents=float(monthly_fee),
        signup_credit_cents=float(signup_credit_dollars) * 100.0,
    )

    has_unsaved_changes = _plan_to_dict(preview_plan) != _plan_to_dict(p)
    if has_unsaved_changes:
        st.warning("Unsaved changes detected for the selected plan.")
    else:
        st.success("No unsaved changes.")
    if final_name != requested_name:
        st.info(f"On save, this plan will be renamed to '{final_name}' to avoid duplicate names.")

    st.divider()
    save_clicked = st.button(
        "Save changes",
        type="primary",
        use_container_width=True,
        key=_ek("save_btn"),
        disabled=not (tou_can_save and fit_tou_can_save),
    )

    if save_clicked:
        old_name = p.name
        plans_lib[selected_idx] = preview_plan

        if st.session_state.get("current_retailer") == old_name:
            st.session_state["_pending_current_retailer"] = final_name
        if st.session_state.get("invoice_plan") == old_name:
            st.session_state["_pending_invoice_plan"] = final_name

        save_plans(plans_lib)
        st.session_state["force_select_plan_idx"] = selected_idx
        if final_name != requested_name:
            st.success(f"Saved. Name '{requested_name}' already existed, so this was saved as '{final_name}'.")
        else:
            st.success("Saved to plans.json.")
        st.rerun()

    st.divider()

    st.markdown("### Import / Export")
    export_json = json.dumps([_plan_to_dict(x) for x in plans_lib], indent=2)
    st.download_button("Download plans.json", export_json, file_name="plans.json", mime="application/json")

    uploaded_plans = st.file_uploader("Upload plans.json to replace library", type=["json"], key="plans_upload")
    if uploaded_plans is not None:
        try:
            data = json.loads(uploaded_plans.getvalue().decode("utf-8"))
            imported = [_dict_to_plan(x) for x in data]
            st.session_state["plans_lib"] = imported
            save_plans(imported)
            st.session_state["force_select_plan_idx"] = 0
            st.success("Imported and saved.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not import: {e}")

    st.divider()


