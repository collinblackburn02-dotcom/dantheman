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
            .attr-card {{ background-color: var(--card); border-radius: 12px; padding: 15px; border: 1px solid rgba(58,42,38,0.1); margin-bottom: 12px; }}
            .attr-title {{ font-weight: 800; color: {BRAND["accent"]}; font-size: 0.95rem; margin-bottom: 8px; }}
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
    # No more parsing needed! Columns are now native to the table.
    df = get_bq_client().query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()
    return df.fillna("") # Replace NaNs with empty strings for clean display

df_master = load_data()

# ================ Sidebar & Controls =================
with st.sidebar:
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric", ["Conv %", "Purchases", "Visitors"])
    min_visitors = st.number_input("Traffic Floor", value=10)

st.title("🪷 Segment Architect")

# Cards for controls
cols = st.columns(3)
configs = [
    ("Gender", "gender"), ("Age", "age"), ("Income", "income"),
    ("State", "state"), ("Net Worth", "net_worth"), ("Children", "children")
]

selected_filters = {}
included_types = []
included_columns = []

for i, (label, col_name) in enumerate(configs):
    with cols[i % 3]:
        st.markdown(f'<div class="attr-card">', unsafe_allow_html=True)
        c_title, c_inc = st.columns([3, 1])
        c_title.markdown(f'<p class="attr-title">{label}</p>', unsafe_allow_html=True)
        
        is_inc = c_inc.checkbox("Inc", value=(i<3), key=f"inc_{col_name}")
        
        valid_opts = sorted([x for x in df_master[col_name].unique() if x != ""])
        val = st.selectbox(f"Filter {label}", ["- All -"] + valid_opts, key=f"filter_{col_name}", label_visibility="collapsed")
        
        if is_inc: 
            included_types.append(col_name)
            included_columns.append(col_name)
        if val != "- All -": 
            selected_filters[col_name] = val
            
        st.markdown('</div>', unsafe_allow_html=True)

# ================ Filtering Logic =================
# We filter by the 'cluster_type' calculated in SQL
active_types = [t for t in included_types]
# Manual combo mapping
if "gender" in included_types and "age" in included_types: active_types.append("gender_age")
if "gender" in included_types and "income" in included_types: active_types.append("gender_income")
if "gender" in included_types and "net_worth" in included_types: active_types.append("gender_nw")
if "gender" in included_types and "children" in included_types: active_types.append("gender_children")
if "gender" in included_types and "age" in included_types and "income" in included_types: active_types.append("gender_age_income")

dff = df_master[df_master['cluster_type'].isin(active_types)].copy()

# Apply isolation filters
for col, val in selected_filters.items():
    dff = dff[dff[col] == val]

# ================ Display =================
if dff.empty:
    st.warning("No data found for this specific target.")
else:
    dff['Conv %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]
    
    metric_map = {"Conv %": "Conv %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    # Show only included columns
    final_cols = included_columns + ["total_visitors", "total_purchasers", "Conv %"]
    
    st.dataframe(
        dff[final_cols].rename(columns={"total_visitors": "Visitors", "total_purchasers": "Purchases"}),
        use_container_width=True
    )
