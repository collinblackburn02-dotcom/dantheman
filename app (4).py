import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors

# ================ 1. CONFIGURATION & THEME =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#B3845C" 

AWS_COLUMN_MAPPER = {
    "GENDER": "gender", "MARRIED": "marital_status", "AGE_RANGE": "age",
    "INCOME_RANGE": "income", "PERSONAL_STATE": "state_raw",
    "PERSONAL_ZIP": "zip_code", "NET_WORTH": "net_worth",
    "SKIPTRACE_CREDIT_RATING": "credit_rating"
}

STATE_TO_REGION = {
    'CT':'Northeast','MA':'Northeast','ME':'Northeast','NH':'Northeast','NJ':'Northeast','NY':'Northeast','PA':'Northeast','RI':'Northeast','VT':'Northeast',
    'IA':'Midwest','IL':'Midwest','IN':'Midwest','KS':'Midwest','MI':'Midwest','MN':'Midwest','MO':'Midwest','ND':'Midwest','NE':'Midwest','OH':'Midwest','SD':'Midwest','WI':'Midwest',
    'AL':'South','AR':'South','DC':'South','DE':'South','FL':'South','GA':'South','KY':'South','LA':'South','MD':'South','MS':'South','NC':'South','OK':'South','SC':'South','TN':'South','TX':'South','VA':'South','WV':'South',
    'AK':'West','AZ':'West','CA':'West','CO':'West','HI':'West','ID':'West','MT':'West','NM':'West','NV':'West','OR':'West','UT':'West','WA':'West','WY':'West'
}

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Audience Engine", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 600 !important; }}
            [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
            div[data-testid="stButton"] button {{ border-radius: 8px; font-weight: 500; padding: 0px 10px !important; }}
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; }}
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px; text-align: center; }}
            [data-testid="stMetricDelta"] svg {{ display: none !important; }}
            [data-testid="stMetricDelta"] div {{ margin-left: 0 !important; }}
            .premium-table-container {{ border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; overflow: hidden; margin-top: 1rem; margin-bottom: 2rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; font-weight: 700 !important; text-align: center !important; padding: 12px !important; border-bottom: 2px solid #D5C6B3 !important; text-transform: uppercase !important; font-size: 0.75rem !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 12px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.85rem !important; }}
            .premium-table-container td:first-child {{ font-weight: 700 !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

# ================ 2. DATA ENGINE =================
@st.cache_data(ttl=3600, show_spinner=False) 
def load_master_graph():
    aws_keys = {"key": st.secrets["aws"]["access_key"], "secret": st.secrets["aws"]["secret_key"], "client_kwargs": {"region_name": "us-east-2"}}
    files = ["master_data.csv", "visitor_data_2.csv"] 
    dataframes = []
    try:
        for f in files:
            path = f"s3://leadnav-demo-data/{f}"
            temp_df = pd.read_csv(path, storage_options=aws_keys, low_memory=False, encoding='latin1', on_bad_lines='skip')
            temp_df.columns = [c.upper().strip() for c in temp_df.columns]
            e_col = 'PERSONAL_EMAILS' if 'PERSONAL_EMAILS' in temp_df.columns else temp_df.columns[0]
            temp_df = temp_df.rename(columns={e_col: 'email_match'})
            dataframes.append(temp_df)
        df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
        df = df.rename(columns=AWS_COLUMN_MAPPER)
        df.columns = [c.lower() for c in df.columns]
        if 'state_raw' in df.columns: df['region'] = df['state_raw'].str.strip().str.upper().map(STATE_TO_REGION).fillna('Unknown')
        if 'gender' in df.columns: df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')
        if 'marital_status' in df.columns: df['marital_status'] = df['marital_status'].map({'Y': 'Married', 'N': 'Single'}).fillna('Unknown')
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
        df['email_match'] = df['email_match'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.split(',')
        df = df.explode('email_match').reset_index(drop=True)
        return df.drop_duplicates(subset=['email_match']).reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 Connection Issue: {e}"); st.stop()

# ================ 3. APP FLOW =================
if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"

if st.session_state.app_state == "onboarding":
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-top: 50px;'>🎯 Audience Engine</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])
        if uploaded_file:
            df_orders = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
            df_orders = df_orders.rename(columns={'Email': 'email_match', 'Name': 'Order ID', 'Total': 'revenue_raw'})
            with st.spinner("Identifying Your Customer Insights..."):
                df_master = load_master_graph()
                df_orders['email_match'] = df_orders['email_match'].astype(str).str.lower().str.strip()
                df_joined = pd.merge(df_orders, df_master, on='email_match', how='inner').reset_index(drop=True)
                df_joined['revenue_raw'] = pd.to_numeric(df_joined['revenue_raw'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)
                
                # 🚨 PRE-CALCULATE EVERYTHING NOW
                all_views = {}
                summary_vars = [("Gender", "gender"), ("Age", "age"), ("Marital Status", "marital_status"), ("Region", "region"), ("State", "state_raw"), ("Zip Code", "zip_code"), ("Credit Rating", "credit_rating")]
                total_rev_all = df_joined['revenue_raw'].sum()
                top_perf = {}

                for label, col_key in summary_vars:
                    temp = df_joined[~df_joined[col_key].astype(str).str.lower().isin(['unknown', 'nan', 'u', 'none', '00nan'])]
                    if not temp.empty:
                        # Summary Data
                        rs = temp.groupby(col_key)['revenue_raw'].sum()
                        top_perf[label] = (rs.idxmax(), (rs.max()/total_rev_all*100))
                        # Table Data
                        grp = temp.groupby(col_key).agg(Purchasers=('Order ID', 'nunique'), Revenue=('revenue_raw', 'sum')).reset_index()
                        grp['% of Buyers'] = (grp['Purchasers'] / grp['Purchasers'].sum()) * 100
                        grp['Rev / Purchaser'] = (grp['Revenue'] / grp['Purchasers'])
                        final_v = grp.rename(columns={col_key: label.upper()}).sort_values('Revenue', ascending=False)
                        if label == "Zip Code": final_v = final_v.head(50)
                        
                        # Store pre-styled HTML
                        all_views[label] = final_v.style.format({'Purchasers': '{:,.0f}', 'Revenue': '${:,.2f}', '% of Buyers': '{:.1f}%', 'Rev / Purchaser': '${:,.2f}'}).background_gradient(subset=['Revenue', '% of Buyers'], cmap=custom_light_green).hide(axis="index").to_html()

                st.session_state.all_precalculated_html = all_views
                st.session_state.top_performers = top_perf
                st.session_state.total_profiles = df_joined['Order ID'].nunique()
                st.session_state.total_revenue = total_rev_all
                st.session_state.app_state = "dashboard"
                st.rerun()

elif st.session_state.app_state == "dashboard":
    if st.button("🔄 New Analysis"): 
        st.session_state.app_state = "onboarding"; st.rerun()

    # 1. MACRO METRICS (Pre-calculated)
    m1, m2 = st.columns(2)
    m1.metric("Resolved Profiles", f"{st.session_state.total_profiles:,.0f}")
    m2.metric("Attributed Sales", f"${st.session_state.total_revenue:,.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

    # 2. TOP PERFORMING DEMOGRAPHICS (Pre-calculated)
    st.markdown("### 🏆 Top Performing Demographics")
    summary_cols = st.columns(len(st.session_state.top_performers))
    for i, (label, data) in enumerate(st.session_state.top_performers.items()):
        summary_cols[i].metric(label, data[0], f"{data[1]:.1f}% of Revenue")
    st.markdown("<hr>", unsafe_allow_html=True)

    # 3. DEEP DIVE SELECTION
    st.markdown("### 🔍 Single Variable Deep Dive")
    if "active_var" not in st.session_state: st.session_state.active_var = "Gender"
    if "active_loc_level" not in st.session_state: st.session_state.active_loc_level = "Region"
    
    # 🚨 Variable Buttons
    v_labels = ["Gender", "Age", "Location", "Marital Status", "Credit Rating"]
    var_cols = st.columns(len(v_labels))
    for i, label in enumerate(v_labels):
        if var_cols[i].button(label, key=f"btn_{label}", type="primary" if st.session_state.active_var == label else "secondary", use_container_width=True):
            st.session_state.active_var = label; st.rerun()

    lookup_key = st.session_state.active_var
    if st.session_state.active_var == "Location":
        st.markdown("<br>", unsafe_allow_html=True)
        l1, l2, l3, _ = st.columns([1, 1, 1, 5])
        if l1.button("Region", type="primary" if st.session_state.active_loc_level == "Region" else "secondary"): st.session_state.active_loc_level = "Region"; st.rerun()
        if l2.button("State", type="primary" if st.session_state.active_loc_level == "State" else "secondary"): st.session_state.active_loc_level = "State"; st.rerun()
        if l3.button("Zip Code", type="primary" if st.session_state.active_loc_level == "Zip Code" else "secondary"): st.session_state.active_loc_level = "Zip Code"; st.rerun()
        lookup_key = st.session_state.active_loc_level

    # 🚨 INSTANT RENDER: Grabbing pre-calculated HTML
    if lookup_key in st.session_state.all_precalculated_html:
        st.markdown(f'<div class="premium-table-container">{st.session_state.all_precalculated_html[lookup_key]}</div>', unsafe_allow_html=True)
