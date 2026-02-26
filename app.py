import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
BRAND = {
    "bg": "#F5F0E6", 
    "fg": "#3A2A26", 
    "accent": "#6E4F3A", 
    "card": "#FFF9F0",
    "gradient_end": "#FDFBF7"
}

def inject_css():
    st.markdown(f"""
        <style>
            .stApp {{ background: var(--bg); color: var(--fg); }}
            .stDataFrame {{ border-radius: 12px; background: var(--card); border: 1px solid rgba(58,42,38,0.1); }}
            .attr-card {{
                background-color: var(--card);
                border-radius: 12px;
                padding: 15px;
                border: 1px solid rgba(58,42,38,0.1);
                margin-bottom: 12px;
            }}
            .attr-title {{ font-weight: 800; color: {BRAND["accent"]}; font-size: 0.95rem; margin-bottom: 8px; }}
            h1, h2, h3 {{ color: {BRAND["fg"]}; font-weight: 800; }}
            /* Checkbox styling */
            .stCheckbox label {{ font-size: 0.85rem; font-weight: 600; color: {BRAND["fg"]}; }}
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
    df = get_bq_client().query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()
    
    # Parse persona string into clean columns, use empty strings instead of 'All'
    def parse_persona(row):
        parts = str(row['demographic_cluster']).split(" + ")
        res = {"Gender": "", "Age": "", "Income": "", "State": "", "NW": "", "Children": ""}
        for p in parts:
            if p in ["M", "F"]: res["Gender"] = p
            elif "-" in p or "older" in p: res["Age"] = p
            elif "$" in p: res["Income"] = p
            elif len(p) == 2 and p.isupper(): res["State"] = p
            elif "Million" in p or "Worth" in p: res["NW"] = p
            elif "Children" in p: res["Children"] = p
        return pd.Series(res)

    parsed = df.apply(parse_persona, axis=1)
    return pd.concat([df, parsed], axis=1)

df_master = load_data()

# ================ Sidebar =================
with st.sidebar:
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric", ["Conv %", "Purchases", "Visitors"])
    min_visitors = st.number_input("Traffic Floor (Min Visitors)", value=10)
    st.markdown("---")
    if st.button("Reset Dashboard"):
        st.rerun()

# ================ Card-Style Control Area =================
st.title("🪷 Segment Architect")
st.markdown("Isolate groups via **Dropdown** or build combos using the **Inc** checkbox.")

# Setup 3-column layout for attributes
cols = st.columns(3)
configs = [
    ("Gender", "gender"), ("Age", "age"), ("Income", "income"),
    ("State", "state"), ("NW", "net_worth"), ("Children", "children")
]

selected_filters = {}
included_labels = []
included_types = []

for i, (label, bq_type) in enumerate(configs):
    with cols[i % 3]:
        st.markdown(f'<div class="attr-card">', unsafe_allow_html=True)
        c_title, c_inc = st.columns([3, 1])
        c_title.markdown(f'<p class="attr-title">{label}</p>', unsafe_allow_html=True)
        
        # Checkbox for column inclusion
        is_inc = c_inc.checkbox("Inc", value=(i<3), key=f"inc_{bq_type}")
        
        # Dropdown for row isolation
        # Filter out empty strings from options
        valid_opts = sorted([x for x in df_master[label].unique() if x != ""])
        val = st.selectbox(f"Filter {label}", ["- All -"] + valid_opts, key=f"filter_{bq_type}", label_visibility="collapsed")
        
        if is_inc: 
            included_labels.append(label)
            included_types.append(bq_type)
        if val != "- All -": 
            selected_filters[label] = val
            
        st.markdown('</div>', unsafe_allow_html=True)

# ================ Combined Filtering Logic =================
# 1. Map the cluster types based on checkboxes
active_types = [t for t in included_types]
# Manual combo mapping for the leaderboard slices
if "gender" in included_types and "age" in included_types: active_types.append("gender_age")
if "gender" in included_types and "income" in included_types: active_types.append("gender_income")
if "gender" in included_types and "state" in included_types: active_types.append("gender_state")
if "gender" in included_types and "age" in included_types and "income" in included_types: active_types.append("gender_age_income")
if "gender" in included_types and "net_worth" in included_types and "income" in included_types: active_types.append("gender_nw_income")
if "gender" in included_types and "state" in included_types and "income" in included_types: active_types.append("gender_state_income")
if "gender" in included_types and "age" in included_types and "children" in included_types: active_types.append("gender_age_children")
if "age" in included_types and "income" in included_types and "net_worth" in included_types: active_types.append("age_income_nw")

dff = df_master[df_master['cluster_type'].isin(active_types)].copy()

# 2. Filter rows based on dropdown selection
for col, val in selected_filters.items():
    dff = dff[dff[col] == val]

# ================ Final Calculations & Sorting =================
if dff.empty:
    st.warning("No data found for this specific target. Try broadening your 'Filter' selection.")
else:
    dff['Conv %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]
    
    # Sorting
    metric_map = {"Conv %": "Conv %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    # 3. SELECT ONLY INCLUDED COLUMNS
    # This makes the columns disappear if 'Inc' is not checked
    final_cols = included_labels + ["total_visitors", "total_purchasers", "Conv %"]
    
    st.dataframe(
        dff[final_cols].rename(columns={"total_visitors": "Visitors", "total_purchasers": "Purchases"})
        .style.format({'Conv %': '{:.2f}%'})
        .background_gradient(subset=['Conv %'], cmap='YlGn'),
        use_container_width=True
    )
