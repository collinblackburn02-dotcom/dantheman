import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
BRAND = {
    "bg": "#F5F0E6",        # beige background
    "fg": "#3A2A26",        # deep brown text
    "accent": "#6E4F3A",    # medium brown
    "accent2": "#A07A5A",   # lighter brown
    "card": "#FFF9F0",      # light card beige
    "white": "#FFFFFF",
}

def inject_css():
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {BRAND["bg"]};
                --fg: {BRAND["fg"]};
                --accent: {BRAND["accent"]};
                --accent2: {BRAND["accent2"]};
                --card: {BRAND["card"]};
                --white: {BRAND["white"]};
            }}
            .stApp {{ background: var(--bg); color: var(--fg); }}
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--card), var(--bg));
                border-right: 1px solid rgba(58,42,38,0.08);
            }}
            .stDataFrame {{ border: 1px solid rgba(58,42,38,0.08); border-radius: 12px; background: var(--card); }}
            .heavenly-section-title {{ font-size: 1.6rem; font-weight: 800; color: var(--fg); margin: 1.5rem 0; }}
            .heavenly-attr-title {{ font-size: 1.25rem; font-weight: 700; color: var(--fg); margin-top: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Heavenly Health | Live Insights", layout="wide")
inject_css()

# ================ BigQuery Connection =================
@st.cache_resource
def get_bq_client():
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    return bigquery.Client(credentials=credentials, project=creds_info["project_id"])

@st.cache_data(ttl=600)
def load_data():
    client = get_bq_client()
    query = "SELECT * FROM `final_dashboard.demographic_leaderboard` WHERE total_purchasers > 0"
    df = client.query(query).to_dataframe()
    return df

# ================ Load Data =================
try:
    df_master = load_data()
except Exception as e:
    st.error(f"Error connecting to BigQuery: {e}")
    st.stop()

# ================ Sidebar =================
with st.sidebar:
    st.markdown("### Controls")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    min_visitors = st.number_input("Minimum Visitors per group", min_value=1, value=20, step=1)

metric_map = {
    "Conversion": "conv_rate",
    "Purchasers": "total_purchasers",
    "Visitors": "total_visitors"
}

# ================ Main Table =================
st.markdown('<div class="heavenly-section-title">Combined Conversion Ranking Table</div>', unsafe_allow_html=True)

# Filter and Sort
dff = df_master[df_master['total_visitors'] >= min_visitors].copy()
dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
dff.index += 1 # Rank starts at 1

# Format for display
disp = dff.copy()
disp.rename(columns={
    "demographic_cluster": "Persona",
    "total_visitors": "Visitors",
    "total_purchasers": "Purchasers",
    "conv_rate": "Conversion %"
}, inplace=True)

# Table Styling
st.dataframe(
    disp.style.format({"Conversion %": "{:.2f}%"})
    .background_gradient(subset=["Conversion %"], cmap='YlGn'),
    use_container_width=True
)

# ================ Summaries =================
st.markdown("---")
st.markdown('<div class="heavenly-section-title">📑 Single-Attribute Summary Tables</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Helper to find single attributes (rows without a ' + ')
def get_singleton(df, label_filter):
    singletons = df[~df['demographic_cluster'].str.contains(r'\+')].copy()
    if label_filter == "Gender":
        return singletons[singletons['demographic_cluster'].isin(['M', 'F'])]
    elif label_filter == "Age":
        # Any singleton that isn't M or F is usually Age or Income
        return singletons[~singletons['demographic_cluster'].isin(['M', 'F'])]
    return singletons

with col1:
    st.markdown('<div class="heavenly-attr-title">Gender</div>', unsafe_allow_html=True)
    st.table(get_singleton(df_master, "Gender")[['demographic_cluster', 'total_visitors', 'total_purchasers', 'conv_rate']])

with col2:
    st.markdown('<div class="heavenly-attr-title">Age / Income</div>', unsafe_allow_html=True)
    st.table(get_singleton(df_master, "Age")[['demographic_cluster', 'total_visitors', 'total_purchasers', 'conv_rate']])
