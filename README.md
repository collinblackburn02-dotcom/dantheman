
# CSV Analytics Dashboard (Streamlit)

This is a **no-code-friendly** Streamlit app: upload a CSV → explore KPIs, filters, and charts → export filtered data.

## Quick Deploy (no local Python required)
1. Create a **new GitHub repo** (public or private).
2. Upload these files to your repo.
3. Go to **Streamlit Community Cloud** → **Deploy an app** → select your repo → set the main file to `app.py`.
4. Click **Deploy**. Done.

## Local Run (optional)
```bash
# 1) Create a virtual environment (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

## CSV Expectations
- Include one column that indicates a **purchase** (e.g., `PURCHASE` as 0/1, or `OrderCount` as an integer).
- Optional columns (auto-detected for filters/charts if present):  
  `GENDER`, `AGE_RANGE`, `INCOME_RANGE`, `NET_WORTH`, `HOMEOWNER`, `SKIPTRACE_CREDIT_RATING`, `MARRIED`, `CHILDREN`, `DATE`

You can select which column is the purchase indicator from the sidebar.

## License
For your commercial use—feel free to white‑label and extend.
