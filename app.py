import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
BRAND = {"bg": "#F5F0E6", "fg": "#3A2A26", "accent": "#6E4F3A", "card": "#FFF9F0"}

def inject_css():
    st.markdown(f"""
        <style>
            .stApp {{ background: var(--bg); color: var(--fg); }}
            .attr-card {{
                background-color: var(--card);
                border-radius: 15px;
                padding: 15px;
                border: 1px solid rgba(58,42,38,0.1);
                margin-bottom: 10px;
            }}
            .attr-title {{ font-weight: 800; font-size: 1rem; color: var(--fg); }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Insights", layout="wide")
inject_css()

# ================ BigQuery Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_data():
    client = get_bq_client()
    return client.query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ================ Sidebar =================
with st.sidebar:
    st.header("Leaderboard Settings")
    metric_choice = st.radio("Sort by:", ["Conversion %", "Purchasers", "Visitors"])
    min_visitors = st.number_input("Min Visitors per segment", value=20)

# ================ Main UI: Toggle Area =================
st.title("🪷 Cumulative Leaderboard")
st.markdown("Select attributes to include their individual and combined stats in the ranking below.")

# Checkboxes for different attributes
col1, col2, col3, col4 = st.columns(4)
with col1: 
    inc_gender = st.checkbox("Gender", value=True)
    inc_children = st.checkbox("Children", value=False)
with col2: 
    inc_age = st.checkbox("Age Range", value=True)
    inc_nw = st.checkbox("Net Worth", value=False)
with col3: 
    inc_income = st.checkbox("Income Range", value=True)
with col4: 
    inc_state = st.checkbox("State", value=False)

# Build the filter list based on checked boxes (Matching BigQuery types)
active_types = []
if inc_gender: active_types.append('gender')
if inc_age: active_types.append('age')
if inc_income: active_types.append('income')
if inc_state: active_types.append('state')
if inc_nw: active_types.append('net_worth')
if inc_children: active_types.append('children')

# Handle Combo Types
if inc_gender and inc_age: active_types.append('gender_age')
if inc_gender and inc_income: active_types.append('gender_income')
if inc_gender and inc_nw: active_types.append('gender_nw')
if inc_gender and inc_children: active_types.append('gender_children')
if inc_gender and inc_age and inc_income: active_types.append('gender_age_income')

# ================ Filter & Display =================
if not active_types:
    st.info("Please check at least one attribute to see the rankings.")
else:
    # Filter the dataframe pulled from BigQuery
    dff = df[df['cluster_type'].isin(active_types)].copy()
    
    # Calculate Final Conversion Metric
    dff['Conversion %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]

    # Map Sidebar Sort Selection
    metric_map = {"Conversion %": "Conversion %", "Purchasers": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1 # Ranks start at 1

    st.subheader("Leaderboard")
    st.dataframe(
        dff[['demographic_cluster', 'total_visitors', 'total_purchasers', 'Conversion %']]
        .rename(columns={
            'demographic_cluster': 'Persona Cluster', 
            'total_visitors': 'Visitors', 
            'total_purchasers': 'Purchases'
        })
        .style.format({'Conversion %': '{:.2f}%'})
        .background_gradient(subset=['Conversion %'], cmap='YlGn'),
        use_container_width=True
    )
