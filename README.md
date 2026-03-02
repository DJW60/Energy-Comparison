# Energex Compare App

Streamlit app for comparing electricity retailer plans using NEM12 interval data.

## Local run

```powershell
pip install -r requirements.txt
python -m streamlit run auto_app_ev.py
```

## Deploy to Streamlit Community Cloud (shareable URL)

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, click **New app**.
3. Choose your repo/branch.
4. Set **Main file path** to `auto_app_ev.py`.
5. Deploy and share the generated URL.

When you update code and push to GitHub, the hosted app can be redeployed so users see updates without installing anything.

## Notes on plan persistence

- The app can read/write `plans.json` on the host instance.
- On Streamlit Cloud, local file storage is not durable long-term.
- Users should use the built-in **Download plans.json** / **Upload plans.json** options for backup and restore.
