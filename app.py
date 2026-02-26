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
            .filter-label {{ font-weight: 700; color: var(--accent); margin-bottom: -15px; }}
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
    # We pull the cumulative table we built in BigQuery
    df = get_bq_client().query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()
    
    # Helper to parse the persona string into actual searchable columns
    def parse_persona(row):
        parts = str(row['demographic_cluster']).split(" + ")
        res = {"Gender": "All", "Age": "All", "Income": "All", "State": "All", "NW": "All", "Children": "All"}
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
    st.header("Global Settings")
    metric_choice = st.radio("Sort leaderboard by:", ["Conversion %", "Purchases", "Visitors"])
    min_visitors = st.number_input("Min Visitors", value=10)

# ================ Multi-Select Filter Area =================
st.title("🪷 Segment Explorer")
st.markdown("Use the dropdowns to isolate specific demographics. Leaving a dropdown at **'All'** includes every sub-group.")

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

# Function to get unique options for dropdowns
def get_opts(col):
    opts = sorted([x for x in df_master[col].unique() if x != "All"])
    return ["All"] + opts

with c1: f_gender = st.selectbox("Gender", get_opts("Gender"))
with c2: f_age = st.selectbox("Age Range", get_opts("Age"))
with c3: f_income = st.selectbox("Income Range", get_opts("Income"))
with c4: f_state = st.selectbox("State", get_opts("State"))
with c5: f_nw = st.selectbox("Net Worth", get_opts("NW"))
with c6: f_children = st.selectbox("Children Status", get_opts("Children"))

# ================ Filtering Logic =================
dff = df_master.copy()

# Apply logic: If user selects 'M', we want any cluster that contains 'M' 
# OR any cluster where Gender is 'All' (to keep single-attribute rows for Age/Income/etc.)
# Actually, the user's intent is usually: "Show me only segments that involve a Male"
filters = {
    "Gender": f_gender, "Age": f_age, "Income": f_income, 
    "State": f_state, "NW": f_nw, "Children": f_children
}

for col, val in filters.items():
    if val != "All":
        dff = dff[dff[col] == val]

# ================ Display =================
if dff.empty:
    st.warning("No clusters found for this specific combination. Try broadening your filters.")
else:
    # Math
    dff['Conversion %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]

    # Sort
    metric_map = {"Conversion %": "Conversion %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    # Final Table View
    display_cols = ["Gender", "Age", "Income", "State", "NW", "Children", "total_visitors", "total_purchasers", "Conversion %"]
    res_table = dff[display_cols].rename(columns={
        "total_visitors": "Visitors", "total_purchasers": "Purchases", "Conversion %": "Conv %"
    })

    st.dataframe(
        res_table.style.format({'Conv %': '{:.2f}%'})
        .background_gradient(subset=['Conv %'], cmap='YlGn'),
        use_container_width=True
    )
