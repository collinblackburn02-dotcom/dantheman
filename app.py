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
            .stDataFrame {{ border-radius: 12px; background: var(--card); }}
            h1, h2, h3 {{ color: var(--fg); font-weight: 800; }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Insights", layout="wide")
inject_css()

# ================ Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_data():
    return get_bq_client().query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()

df = load_data()

# ================ Sidebar =================
with st.sidebar:
    st.header("Settings")
    metric_choice = st.radio("Sort by:", ["Conversion %", "Purchases", "Visitors"])
    min_visitors = st.number_input("Min Visitors", value=20)

# ================ Toggle Area =================
st.title("🪷 Cumulative Leaderboard")
st.markdown("Select attributes to rank individual and combined personas.")

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

# Logic to map checkboxes to BigQuery 'cluster_type'
active_types = []
if inc_gender: active_types.append('gender')
if inc_age: active_types.append('age')
if inc_income: active_types.append('income')
if inc_state: active_types.append('state')
if inc_nw: active_types.append('net_worth')
if inc_children: active_types.append('children')

# Pairs
if inc_gender and inc_age: active_types.append('gender_age')
if inc_gender and inc_income: active_types.append('gender_income')
if inc_gender and inc_state: active_types.append('gender_state')
if inc_gender and inc_nw: active_types.append('gender_nw')
if inc_gender and inc_children: active_types.append('gender_children')
if inc_age and inc_income: active_types.append('age_income')
if inc_state and inc_income: active_types.append('state_income')

# 3-Ways
if inc_gender and inc_age and inc_income: active_types.append('gender_age_income')
if inc_gender and inc_nw and inc_income: active_types.append('gender_nw_income')
if inc_gender and inc_state and inc_income: active_types.append('gender_state_income')
if inc_gender and inc_age and inc_children: active_types.append('gender_age_children')
if inc_age and inc_income and inc_nw: active_types.append('age_income_nw')

# ================ Formatting Logic =================
if not active_types:
    st.info("Check boxes above to see data.")
else:
    dff = df[df['cluster_type'].isin(active_types)].copy()
    
    # SPLIT THE PERSONA STRING BACK INTO COLUMNS
    # This logic looks at the text and puts it in the right "bucket" column
    def parse_persona(row):
        parts = str(row['demographic_cluster']).split(" + ")
        res = {"Gender": "", "Age": "", "Income": "", "State": "", "NW": "", "Children": ""}
        
        for p in parts:
            if p in ["M", "F"]: res["Gender"] = p
            elif "-" in p or "older" in p: res["Age"] = p
            elif "$" in p and ("k" in p or "000" in p): res["Income"] = p
            elif len(p) == 2 and p.isupper(): res["State"] = p
            elif "Million" in p or "Worth" in p: res["NW"] = p
            elif p in ["Has Children", "No Children", "Children"]: res["Children"] = p
        return pd.Series(res)

    parsed_cols = dff.apply(parse_persona, axis=1)
    dff = pd.concat([dff, parsed_cols], axis=1)

    # Metrics
    dff['Conversion %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]

    # Sort
    metric_map = {"Conversion %": "Conversion %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    # Clean up display table
    display_df = dff[["Gender", "Age", "Income", "State", "NW", "Children", "total_visitors", "total_purchasers", "Conversion %"]]
    display_df.columns = ["Gender", "Age Range", "Income", "State", "Net Worth", "Children", "Visitors", "Purchases", "Conv %"]

    st.dataframe(
        display_df.style.format({'Conv %': '{:.2f}%'})
        .background_gradient(subset=['Conv %'], cmap='YlGn'),
        use_container_width=True
    )
