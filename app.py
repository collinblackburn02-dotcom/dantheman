import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os

# ================ Brand palette & CSS =================
BRAND = {
    "bg": "#FAF8F5",       
    "fg": "#3A2A26",       
    "accent": "#8C6239",   
    "card": "#FFFFFF"      
}

def inject_css():
    st.markdown(f"""
        <style>
            .stApp {{ background: {BRAND["bg"]}; color: {BRAND["fg"]}; }}
            .stDataFrame {{ border-radius: 12px; background: {BRAND["card"]}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            .attr-title {{ font-weight: 800; color: {BRAND["accent"]}; font-size: 0.95rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
            [data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid rgba(140, 98, 57, 0.1); }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Insights", page_icon="🪵", layout="wide")
inject_css()

# Look for 'logo.png' in the github folder. If it's there, display it!
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
else:
    st.sidebar.markdown(f"<h2 style='color: {BRAND['accent']};'>🪵 Heavenly Heat</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# ================ Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_data():
    client = get_bq_client()
    df = client.query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()
    return df.fillna("")

df_master = load_data()

# ================ Sidebar & Global Controls =================
with st.sidebar:
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric", ["Conv %", "Purchases", "Visitors"])
    min_visitors = st.number_input("Traffic Floor", value=10)
    st.markdown("---")
    if st.button("Reset Filters"):
        st.rerun()

st.title("🪵 Audience Insights Engine")

# ================ 1. UI: Checkboxes and Dropdowns =================
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
        # Using native Streamlit container to eliminate the white ovals
        with st.container(border=True):
            c_title, c_inc = st.columns([3, 1])
            c_title.markdown(f'<p class="attr-title">{label}</p>', unsafe_allow_html=True)
            
            is_inc = c_inc.checkbox("Inc", value=(i<3), key=f"inc_{col_name}")
            
            valid_opts = sorted([x for x in df_master[col_name].unique() if x != ""])
            val = st.selectbox(f"Filter {label}", ["- All -"] + valid_opts, key=f"filter_{col_name}", label_visibility="collapsed")
            
            if is_inc: 
                included_types.append(col_name)
                included_columns.append(label)
            if val != "- All -": 
                selected_filters[col_name] = val

# ================ 2. Logic: Resolve Combinations =================
active_types = []
if included_types:
    inc_set = set(included_types)
    active_types.extend(included_types) 

    if "gender" in inc_set and "age" in inc_set: active_types.append("gender_age")
    if "gender" in inc_set and "income" in inc_set: active_types.append("gender_income")
    if "gender" in inc_set and "state" in inc_set: active_types.append("gender_state")
    if "gender" in inc_set and "net_worth" in inc_set: active_types.append("gender_nw")
    if "gender" in inc_set and "children" in inc_set: active_types.append("gender_children")
    if "age" in inc_set and "income" in inc_set: active_types.append("age_income")
    if "age" in inc_set and "net_worth" in inc_set: active_types.append("age_nw")
    if "state" in inc_set and "income" in inc_set: active_types.append("state_income")
    if "income" in inc_set and "net_worth" in inc_set: active_types.append("income_nw")

    if {"gender", "age", "income"}.issubset(inc_set): active_types.append("gender_age_income")
    if {"gender", "age", "state"}.issubset(inc_set): active_types.append("gender_age_state")
    if {"gender", "income", "state"}.issubset(inc_set): active_types.append("gender_income_state")
    if {"gender", "income", "net_worth"}.issubset(inc_set): active_types.append("gender_income_nw")
    if {"gender", "age", "children"}.issubset(inc_set): active_types.append("gender_age_children")
    if {"age", "income", "net_worth"}.issubset(inc_set): active_types.append("age_income_nw")
    if {"state", "income", "net_worth"}.issubset(inc_set): active_types.append("state_income_nw")
    if {"gender", "state", "net_worth"}.issubset(inc_set): active_types.append("gender_state_nw")

# ================ 3. Filtering and Display =================
dff = df_master[df_master['cluster_type'].isin(active_types)].copy()

for col, val in selected_filters.items():
    dff = dff[dff[col] == val]

if dff.empty:
    st.warning("No data found for this combination.")
else:
    dff['Conv %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]
    
    # ------------------ SKU Parsing Logic ------------------
    sku_cols = []
    if 'sku_string' in dff.columns:
        # Parse the aggregated BigQuery string into a dynamic dictionary of products
        parsed_skus = dff['sku_string'].apply(
            lambda x: dict(item.split("::") for item in str(x).split("~~") if "::" in item) if pd.notna(x) and x != "" else {}
        )
        # Turn the dictionary into beautiful, sortable columns
        sku_df = pd.DataFrame(parsed_skus.tolist(), index=dff.index).fillna(0).astype(int)
        dff = pd.concat([dff, sku_df], axis=1)
        sku_cols = sorted(list(sku_df.columns))
    # -------------------------------------------------------

    metric_map = {"Conv %": "Conv %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    label_to_col = {"Gender": "gender", "Age Range": "age", "Income Range": "income", 
                    "State": "state", "Net Worth": "net_worth", "Children": "children"}
    
    final_display_cols = []
    for label in ["Gender", "Age Range", "Income Range", "State", "Net Worth", "Children"]:
        internal_name = label_to_col.get(label)
        if internal_name in included_types:
            final_display_cols.append(internal_name)
    
    # Append the base metrics, then pin all the new SKU columns to the far right!
    final_display_cols += ["total_visitors", "total_purchasers", "Conv %"] + sku_cols
    
    st.dataframe(
        dff[final_display_cols].rename(columns={
            "gender": "Gender", "age": "Age", "income": "Income", 
            "state": "State", "net_worth": "Net Worth", "children": "Children",
            "total_visitors": "Visitors", "total_purchasers": "Purchases"
        }).style.format({'Conv %': '{:.2f}%'}).background_gradient(subset=['Conv %'], cmap='YlGn'),
        use_container_width=True
    )
