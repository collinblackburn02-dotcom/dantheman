import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
BRAND = {
    "bg": "#F5F0E6", "fg": "#3A2A26", "accent": "#6E4F3A", 
    "accent2": "#A07A5A", "card": "#FFF9F0", "white": "#FFFFFF",
}

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

# ================ BigQuery Connection =================
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
    query = "SELECT * FROM `final_dashboard.demographic_leaderboard` WHERE total_purchasers > 0"
    return client.query(query).to_dataframe()

try:
    df_master = load_data()
except Exception as e:
    st.error(f"Error connecting to BigQuery: {e}")
    st.stop()

# ================ Sidebar Controls =================
with st.sidebar:
    st.markdown("### Controls")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    min_visitors = st.number_input("Minimum Visitors per group", min_value=1, value=10, step=1)
    
    st.markdown("---")
    st.markdown("### Toggle Variables")
    options = ["Gender", "Age", "Income", "Homeowner/Renter", "State"]
    selected_vars = st.multiselect("Show clusters containing:", options, default=options)

# ================ Logic to Filter Variables & Remove "U" =================
keyword_map = {
    "Gender": ["M", "F"],
    "Age": ["18-24", "25-34", "35-44", "45-54", "55-64", "65 and older"],
    "Income": ["$", "k", "000"],
    "Homeowner/Renter": ["Homeowner", "Renter"],
    "State": [s for s in df_master['demographic_cluster'].unique() if len(str(s)) == 2 and str(s).isupper()]
}

excluded_vars = [v for v in options if v not in selected_vars]
keywords_to_remove = []
for v in excluded_vars:
    keywords_to_remove.extend(keyword_map[v])

# Add "U" to the permanent removal list
permanent_removal = ["U", "Unknown", "U +", "+ U"]

def filter_clusters(row):
    cluster_str = str(row['demographic_cluster'])
    
    # 1. Permanently remove "U" gender clusters
    if any(u_val in cluster_str.split(" + ") for u_val in ["U", "Unknown"]):
        return False
        
    # 2. Apply the dynamic sidebar toggles
    for word in keywords_to_remove:
        if word in cluster_str:
            return False
    return True

dff = df_master[df_master.apply(filter_clusters, axis=1)].copy()

# ================ Leaderboard Display =================
st.markdown('<div class="heavenly-section-title">🪷 Combined Conversion Ranking Table</div>', unsafe_allow_html=True)

metric_map = {"Conversion": "conv_rate", "Purchasers": "total_purchasers", "Visitors": "total_visitors"}

dff = dff[dff['total_visitors'] >= min_visitors]
dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
dff.index += 1

disp = dff.rename(columns={
    "demographic_cluster": "Persona Cluster",
    "total_visitors": "Visitors",
    "total_purchasers": "Purchases",
    "conv_rate": "Conversion %"
})

st.dataframe(
    disp.style.format({"Conversion %": "{:.2f}%"})
    .background_gradient(subset=["Conversion %"], cmap='YlGn'),
    use_container_width=True
)

st.info(f"Displaying {len(dff)} high-value clusters.")
st.info(f"Displaying {len(dff)} clusters.")
