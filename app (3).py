import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import os
import altair as alt

# ================ 0. PITCH CONFIGURATION =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#0A2540" 

# 🚨 THE DATA MAPPER
AWS_COLUMN_MAPPER = {
    "GENDER": "gender",
    "AGE_RANGE": "age",
    "INCOME_RANGE": "income_raw",
    "PERSONAL_STATE": "state_raw",
    "NET_WORTH": "net_worth_raw",
    "CHILDREN": "children",
    "MARRIED": "marital_status",
    "HOMEOWNER": "homeowner",
    "SKIPTRACE_CREDIT_RATING": "credit_raw"
}

# 🗺️ REGIONAL MAPPING
STATE_TO_REGION = {
    'CT':'Northeast','MA':'Northeast','ME':'Northeast','NH':'Northeast','NJ':'Northeast','NY':'Northeast','PA':'Northeast','RI':'Northeast','VT':'Northeast',
    'IA':'Midwest','IL':'Midwest','IN':'Midwest','KS':'Midwest','MI':'Midwest','MN':'Midwest','MO':'Midwest','ND':'Midwest','NE':'Midwest','OH':'Midwest','SD':'Midwest','WI':'Midwest',
    'AL':'South','AR':'South','DC':'South','DE':'South','FL':'South','GA':'South','KY':'South','LA':'South','MD':'South','MS':'South','NC':'South','OK':'South','SC':'South','TN':'South','TX':'South','VA':'South','WV':'South',
    'AK':'West','AZ':'West','CA':'West','CO':'West','HI':'West','ID':'West','MT':'West','NM':'West','NV':'West','OR':'West','UT':'West','WA':'West','WY':'West'
}

# =========================================================

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Customer DNA", page_icon="🧬", layout="centered")

# INITIALIZE SESSION STATE
if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"
if "df_icp" not in st.session_state: st.session_state.df_icp = None

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 700 !important; }}
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px; }}
            .premium-table-container {{ width: 100%; border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; margin-bottom: 5rem; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; text-transform: uppercase; font-size: 0.75rem; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#6EAB6E"])

def render_premium_table(styler_obj):
    st.markdown(f'<div class="premium-table-container">{styler_obj.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# BUCKETING HELPERS
def bucket_income(val):
    v = str(val).lower()
    if any(x in v for x in ['250', '500']): return "High Earner ($250k+)"
    if any(x in v for x in ['125', '150', '175', '200']): return "Upper Middle ($125k-$249k)"
    if any(x in v for x in ['50', '75', '100']): return "Middle Class ($50k-$124k)"
    return "Emerging (Under $50k)"

def bucket_nw(val):
    v = str(val).lower()
    if any(x in v for x in ['1,000', '2,000', '5,000']): return "High Net Worth ($1M+)"
    if any(x in v for x in ['250', '500']): return "Mass Affluent ($250k-$999k)"
    if any(x in v for x in ['50', '100']): return "Emerging Wealth ($50k-$249k)"
    return "Entry Level (Under $50k)"

def bucket_credit(val):
    v = str(val).upper().strip()
    if v in ['A', 'B', 'C']: return "Elite/Prime (A, B, C)"
    if v in ['D', 'E']: return "Standard (D, E)"
    if v in ['F', 'G']: return "Developing (F, G)"
    return "U"

# ================ 2. LIVE AWS CONNECTION =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    aws_keys = {"key": st.secrets["aws"]["access_key"], "secret": st.secrets["aws"]["secret_key"], "client_kwargs": {"region_name": "us-east-2"}}
    try:
        df = pd.read_csv("s3://leadnav-demo-data/master_data.csv", storage_options=aws_keys, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        
        # 1. First Index Reset
        df = df.reset_index(drop=True)
        
        rename_dict = {k.lower(): v for k, v in AWS_COLUMN_MAPPER.items()}
        df = df.rename(columns=rename_dict)
        
        if 'state_raw' in df.columns: df['region'] = df['state_raw'].str.strip().str.upper().map(STATE_TO_REGION)
        if 'income_raw' in df.columns: df['income'] = df['income_raw'].apply(bucket_income)
        if 'net_worth_raw' in df.columns: df['net_worth'] = df['net_worth_raw'].apply(bucket_nw)
        if 'credit_raw' in df.columns: df['credit_rating'] = df['credit_raw'].apply(bucket_credit)
        
        # Email Explosion logic
        email_col = next((c for c in df.columns if 'email' in c.lower()), 'Email')
        df = df.rename(columns={email_col: 'Email'})
        df['Email'] = df['Email'].astype(str).str.lower().str.split(',')
        
        # 2. Explode creates duplicate index labels
        df = df.explode('Email')
        df['Email'] = df['Email'].str.strip()
        
        # 3. CRITICAL: Reset Index after explosion to kill the 'Duplicate Labels' error
        df = df.drop_duplicates(subset=['Email'], keep='first').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"🚨 AWS Error: {e}"); st.stop()

# ================ 3. DASHBOARD LOGIC =================
if st.session_state.app_state == "onboarding":
    st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem;'>{PITCH_COMPANY_NAME}</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Customer File", type=["csv"])
    if uploaded_file:
        df_orders = pd.read_csv(uploaded_file)
        df_orders = df_orders.rename(columns={'Name': 'Order ID', 'Created at': 'Date'})
        with st.spinner("Analyzing Identity Graph..."):
            df_master = load_master_graph()
            df_orders['Email'] = df_orders['Email'].astype(str).str.lower().str.strip()
            
            # Merge logic
            df_joined = pd.merge(df_orders, df_master, on='Email', how='inner').reset_index(drop=True)
            
            if not df_joined.empty:
                st.session_state.df_icp = df_joined
                st.session_state.app_state = "dashboard"
                st.rerun()

elif st.session_state.app_state == "dashboard":
    st.title("🧬 Customer DNA Analysis")
    if st.button("← New Upload"): st.session_state.app_state = "onboarding"; st.rerun()
    
    df = st.session_state.df_icp
    df['Total'] = pd.to_numeric(df['Total'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)

    c1, c2 = st.columns(2)
    c1.metric("Resolved Profiles", f"{df['Order ID'].nunique():,.0f}")
    c2.metric("Attributed Sales", f"${df['Total'].sum():,.2f}")

    display_configs = [
        ("Credit Rating Score", "credit_rating"), ("Household Income", "income"), ("Net Worth Segment", "net_worth"), 
        ("Regional Footprint", "region"), ("Age Range", "age"), ("Gender", "gender"), ("Marital Status", "marital_status")
    ]

    for label, col in display_configs:
        if col in df.columns:
            df_plot = df[~df[col].astype(str).str.lower().isin(['u', 'unknown', 'nan', 'none', '', 'other'])]
            grp = df_plot.groupby(col).agg(Buyers=('Order ID', 'nunique'), Revenue=('Total', 'sum')).reset_index()
            
            if not grp.empty:
                # Force numeric before percentage math
                grp['Revenue'] = pd.to_numeric(grp['Revenue'], errors='coerce').fillna(0)
                
                st.markdown(f"## {label}")
                if col == "region":
                    with st.expander("📍 View Regional Identity Map"):
                        st.write("**Northeast:** CT, MA, ME, NH, NJ, NY, PA, RI, VT")
                        st.write("**Midwest:** IA, IL, IN, KS, MI, MN, MO, ND, NE, OH, SD, WI")
                        st.write("**South:** AL, AR, DC, DE, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV")
                        st.write("**West:** AK, AZ, CA, CO, HI, ID, MT, NM, NV, OR, UT, WA, WY")

                chart = alt.Chart(grp).mark_arc(innerRadius=80, stroke="#fff").encode(
                    theta="Revenue:Q", color=alt.Color(f"{col}:N", scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(title=None, orient="bottom", columns=2)),
                    tooltip=[col, alt.Tooltip('Revenue', format='$,.0f')]
                ).properties(height=500)
                st.altair_chart(chart, use_container_width=True)
                
                grp['% Share'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col: label})
                render_premium_table(grp.style.format({'Buyers': '{:,.0f}', 'Revenue': '${:,.2f}', '% Share': '{:.1f}%', 'AOV': '${:,.2f}'}))
