import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
# Inspired by Heavenly Heat Saunas: Clean whites, warm hemlock woods, and dark brown text
BRAND = {
    "bg": "#FAF8F5",       # Very light, warm off-white background
    "fg": "#3A2A26",       # Dark brown for main text (matches your logo text)
    "accent": "#8C6239",   # Warm cedar/hemlock brown for accents and titles
    "card": "#FFFFFF"      # Crisp white for the data cards
}

def inject_css():
    st.markdown(f"""
        <style>
            /* Main background and text */
            .stApp {{ background: {BRAND["bg"]}; color: {BRAND["fg"]}; }}
            
            /* Dataframe styling */
            .stDataFrame {{ border-radius: 12px; background: {BRAND["card"]}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            
            /* Custom attribute cards */
            .attr-card {{ 
                background-color: {BRAND["card"]}; 
                border-radius: 12px; 
                padding: 15px; 
                border: 1px solid rgba(140, 98, 57, 0.2); 
                margin-bottom: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }}
            .attr-title {{ font-weight: 800; color: {BRAND["accent"]}; font-size: 0.95rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid rgba(140, 98, 57, 0.1); }}
        </style>
        """, unsafe_allow_html=True)

# Set the page title and icon
st.set_page_config(page_title="Heavenly Insights", page_icon="🪵", layout="wide")
inject_css()

# Inject the logo at the top of the sidebar
# (If the image link breaks in the future, just right-click your logo on your website, click "Copy Image Address", and paste it here!)
st.sidebar.image("https://heavenlyheatsaunas.com/cdn/shop/files/Heavenly_Heat_Saunas_Logo.png", use_container_width=True)
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

st.title("🪷 Segment Architect")

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
        st.markdown(f'<div class="attr-card">', unsafe_allow_html=True)
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
            
        st.markdown('</div>', unsafe_allow_html=True)

# ================ 2. Logic: Resolve Combinations =================
active_types = []
if included_types:
    inc_set = set(included_types)
    active_types.extend(included_types) # Singles

    # Two-Way Logic
    if "gender" in inc_set and "age" in inc_set: active_types.append("gender_age")
    if "gender" in inc_set and "income" in inc_set: active_types.append("gender_income")
    if "gender" in inc_set and "state" in inc_set: active_types.append("gender_state")
    if "gender" in inc_set and "net_worth" in inc_set: active_types.append("gender_nw")
    if "gender" in inc_set and "children" in inc_set: active_types.append("gender_children")
    if "age" in inc_set and "income" in inc_set: active_types.append("age_income")
    if "age" in inc_set and "net_worth" in inc_set: active_types.append("age_nw")
    if "state" in inc_set and "income" in inc_set: active_types.append("state_income")
    if "income" in inc_set and "net_worth" in inc_set: active_types.append("income_nw")

    # Three-Way Logic
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

# Apply isolation filters from dropdowns
for col, val in selected_filters.items():
    dff = dff[dff[col] == val]

if dff.empty:
    st.warning("No data found for this combination.")
else:
    dff['Conv %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]
    
    metric_map = {"Conv %": "Conv %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    # Map the labels back to internal columns for display
    label_to_col = {"Gender": "gender", "Age Range": "age", "Income Range": "income", 
                    "State": "state", "Net Worth": "net_worth", "Children": "children"}
    
    # Only show the columns the user checked 'Inc' for
    final_display_cols = []
    for label in ["Gender", "Age Range", "Income Range", "State", "Net Worth", "Children"]:
        internal_name = label_to_col.get(label)
        if internal_name in included_types:
            final_display_cols.append(internal_name)
    
    final_display_cols += ["total_visitors", "total_purchasers", "Conv %"]
    
    st.dataframe(
        dff[final_display_cols].rename(columns={
            "gender": "Gender", "age": "Age", "income": "Income", 
            "state": "State", "net_worth": "Net Worth", "children": "Children",
            "total_visitors": "Visitors", "total_purchasers": "Purchases"
        }).style.format({'Conv %': '{:.2f}%'}).background_gradient(subset=['Conv %'], cmap='YlGn'),
        use_container_width=True
    )
