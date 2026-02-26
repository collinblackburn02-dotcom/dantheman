import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
BRAND = {"bg": "#F5F0E6", "fg": "#3A2A26", "accent": "#6E4F3A", "card": "#FFF9F0"}

def inject_css():
    st.markdown(f"""
        <style>
            :root {{ --bg: {BRAND["bg"]}; --fg: {BRAND["fg"]}; --accent: {BRAND["accent"]}; }}
            .stApp {{ background: var(--bg); color: var(--fg); }}
            section[data-testid="stSidebar"] {{ background: var(--card); border-right: 1px solid rgba(58,42,38,0.08); }}
            .stDataFrame {{ border-radius: 12px; background: var(--card); }}
            .heavenly-section-title {{ font-size: 1.6rem; font-weight: 800; color: var(--fg); margin: 1.5rem 0; }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Health | Insights", layout="wide")
inject_css()

# ================ Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    return bigquery.Client(credentials=credentials, project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_data():
    client = get_bq_client()
    return client.query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()

try:
    df_master = load_data()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ================ Sidebar =================
with st.sidebar:
    st.markdown("### Controls")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    min_visitors = st.number_input("Min Visitors", min_value=1, value=5)
    
    st.markdown("---")
    st.markdown("### Toggle Variables")
    
    # NEW: Select categories based on the actual BigQuery column
    all_categories = sorted(df_master['category'].unique())
    selected_categories = st.multiselect(
        "Show these categories:", 
        options=all_categories, 
        default=all_categories
    )

# ================ Filtering =================
# We filter directly on the CATEGORY column. 100% accurate.
dff = df_master[df_master['category'].isin(selected_categories)].copy()

# ================ Display =================
st.markdown('<div class="heavenly-section-title">🪷 Combined Conversion Ranking Table</div>', unsafe_allow_html=True)

metric_map = {"Conversion": "conv_rate", "Purchasers": "total_purchasers", "Visitors": "total_visitors"}
dff = dff[dff['total_visitors'] >= min_visitors].sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
dff.index += 1

st.dataframe(
    dff.rename(columns={
        "demographic_cluster": "Persona Cluster", 
        "category": "Category",
        "total_visitors": "Visitors", 
        "total_purchasers": "Purchases", 
        "conv_rate": "Conversion %"
    })
    .style.format({"Conversion %": "{:.2f}%"})
    .background_gradient(subset=["Conversion %"], cmap='YlGn'),
    use_container_width=True
)
