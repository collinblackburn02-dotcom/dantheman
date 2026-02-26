import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS (STAYING THE SAME) =================
BRAND = {
    "bg": "#F5F0E6",        # beige background
    "fg": "#3A2A26",        # deep brown text
    "accent": "#6E4F3A",    # medium brown
    "accent2": "#A07A5A",   # lighter brown
    "card": "#FFF9F0",      # light card beige
    "white": "#FFFFFF",
}

def inject_css():
    st.markdown(f"""
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
            .stDataFrame {{ border-radius: 12px; background: var(--card); }}
            .heavenly-section-title {{ font-size: 1.6rem; font-weight: 800; color: var(--fg); margin: 1.5rem 0; }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Health | Insights", layout="wide")
inject_css()

# ================ BigQuery Authentication =================
@st.cache_resource
def get_bq_client():
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    return bigquery.Client(credentials=credentials, project=creds_info["project_id"])

client = get_bq_client()

# ================ Data Loading =================
@st.cache_data(ttl=600)
def load_data():
    # This query pulls the clean table we built in BigQuery
    query = "SELECT * FROM `final_dashboard.demographic_leaderboard` WHERE total_purchasers > 0"
    df = client.query(query).to_dataframe()
    return df

df_master = load_data()

# ================ Sidebar Controls =================
with st.sidebar:
    st.markdown("### Controls")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=50, step=1)

metric_map = {
    "Conversion": "conv_rate", 
    "Purchasers": "total_purchasers", 
    "Visitors": "total_visitors"
}

# ================ Main Table Logic =================
st.markdown('<div class="heavenly-section-title">🪷 Combined Conversion Ranking Table</div>', unsafe_allow_html=True)

# Filter by visitors
dff = df_master[df_master['total_visitors'] >= min_rows].copy()
dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
dff.index += 1 # Rank starts at 1

# Formatting for display
disp = dff.copy()
disp['conv_rate'] = disp['conv_rate'].map(lambda x: f"{x:.2f}%")
disp['total_visitors'] = disp['total_visitors'].map(lambda x: f"{int(x):,}")
disp['total_purchasers'] = disp['total_purchasers'].map(lambda x: f"{int(x):,}")

# Table Display
st.dataframe(
    disp.style.background_gradient(subset=[metric_map[metric_choice]], cmap='YlGn'),
    use_container_width=True
)

# ================ Single-Attribute Summaries (The "Pretty" Charts) =================
st.markdown("---")
st.markdown('<div class="heavenly-section-title">📑 Single-Attribute Summaries</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Gender")
    # Rows without ' + ' are single-attribute
    gender_df = df_master[~df_master['demographic_cluster'].str.contains('\+')].copy()
    # Looking for M/F specifically
    gender_df = gender_df[gender_df['demographic_cluster'].str.contains('M|F', na=False)]
    st.dataframe(gender_df[['demographic_cluster', 'total_visitors', 'total_purchasers', 'conv_rate']], hide_index=True)

with col2:
    st.markdown("### Age Range")
    age_df = df_master[~df_master['demographic_cluster'].str.contains('\+')].copy()
    # Exclude M/F to find Age groups
    age_df = age_df[~age_df['demographic_cluster'].str.contains('M|F', na=False)]
    st.dataframe(age_df[['demographic_cluster', 'total_visitors', 'total_purchasers', 'conv_rate']], hide_index=True)
