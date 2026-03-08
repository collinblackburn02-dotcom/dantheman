import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import os
import altair as alt

# ================ 1. PITCH CONFIGURATION =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#0A2540" 

AWS_COLUMN_MAPPER = {
    "GENDER": "gender",
    "MARRIED": "marital_status",
    "AGE_RANGE": "age",
    "INCOME_RANGE": "income_raw",
    "PERSONAL_STATE": "state_raw",
    "NET_WORTH": "net_worth_raw",
    "CHILDREN": "children",
    "HOMEOWNER": "homeowner",
    "SKIPTRACE_CREDIT_RATING": "credit_raw"
}

STATE_TO_REGION = {
    'CT':'Northeast','MA':'Northeast','ME':'Northeast','NH':'Northeast','NJ':'Northeast','NY':'Northeast','PA':'Northeast','RI':'Northeast','VT':'Northeast',
    'IA':'Midwest','IL':'Midwest','IN':'Midwest','KS':'Midwest','MI':'Midwest','MN':'Midwest','MO':'Midwest','ND':'Midwest','NE':'Midwest','OH':'Midwest','SD':'Midwest','WI':'Midwest',
    'AL':'South','AR':'South','DC':'South','DE':'South','FL':'South','GA':'South','KY':'South','LA':'South','MD':'South','MS':'South','NC':'South','OK':'South','SC':'South','TN':'South','TX':'South','VA':'South','WV':'South',
    'AK':'West','AZ':'West','CA':'West','CO':'West','HI':'West','ID':'West','MT':'West','NM':'West','NV':'West','OR':'West','UT':'West','WA':'West','WY':'West'
}

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Customer DNA", page_icon="🧬", layout="centered")

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
            .premium-table-container {{ width: 700px !important; margin: 0 auto 5rem auto; border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; overflow: hidden; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; padding: 12px; text-transform: uppercase; font-size: 0.75rem; }}
            .premium-table-container td {{ text-align: center !important; padding: 10px; border-bottom: 1px solid #F0EAD6; font-size: 0.9rem; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#6EAB6E"])

def render_premium_table(styler_obj):
    st.markdown(f'<div class="premium-table-container">{styler_obj.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# BUCKETING LOGIC
def bucket_income(val):
    v = str(val).lower()
    if any(x in v for x in ['250', '500']): return "High ($250k+)"
    if any(x in v for x in ['125', '150', '175', '200']): return "Med-High ($125k-$249k)"
    if any(x in v for x in ['50', '75', '100']): return "Medium ($50k-$124k)"
    return "Low (Under $50k)"

def bucket_nw(val):
    v = str(val).lower()
    if any(x in v for x in ['1,000', '2,000', '5,000']): return "High ($1M+)"
    if any(x in v for x in ['250', '500']): return "Med-High ($250k-$999k)"
    if any(x in v for x in ['50', '100']): return "Medium ($50k-$249k)"
    return "Low (Under $50k)"

def bucket_credit(val):
    v = str(val).upper().strip()
    if v in ['A', 'B', 'C']: return "High (A, B, C)"
    if v in ['D', 'E']: return "Medium (D, E)"
    if v in ['F', 'G']: return "Low (F, G)"
    return "Unknown"

# ================ 2. LIVE AWS CONNECTION =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    aws_keys = {"key": st.secrets["aws"]["access_key"], "secret": st.secrets["aws"]["secret_key"], "client_kwargs": {"region_name": "us-east-2"}}
    files = ["master_data.csv", "visitor_data_2.csv"] 
    dataframes = []
    
    try:
        for f in files:
            path = f"s3://leadnav-demo-data/{f}"
            # 🚨 latin1 is more permissive for messy CSVs
            temp_df = pd.read_csv(path, storage_options=aws_keys, low_memory=False, encoding='latin1', on_bad_lines='skip')
            dataframes.append(temp_df)
            
        df = pd.concat(dataframes, axis=0, ignore_index=True)
        df.columns = [c.upper() for c in df.columns] # Use Upper for mapping safety
        df = df.reset_index(drop=True)
        
        # Mapping Logic
        df = df.rename(columns=AWS_COLUMN_MAPPER)
        
        if 'STATE_RAW' in df.columns: df['region'] = df['STATE_RAW'].str.strip().str.upper().map(STATE_TO_REGION)
        if 'INCOME_RAW' in df.columns: df['income'] = df['INCOME_RAW'].apply(bucket_income)
        if 'NET_WORTH_RAW' in df.columns: df['net_worth'] = df['NET_WORTH_RAW'].apply(bucket_nw)
        if 'CREDIT_RAW' in df.columns: df['credit_rating'] = df['CREDIT_RAW'].apply(bucket_credit)
        
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')
        if 'marital_status' in df.columns:
            df['marital_status'] = df['marital_status'].map({'Y': 'Married', 'N': 'Single'}).fillna('Unknown')
        
        # 🚨 SUPER-MATCH LOGIC: Prioritize PERSONAL_EMAILS over hashes
        target_email_col = 'PERSONAL_EMAILS' if 'PERSONAL_EMAILS' in df.columns else None
        if not target_email_col:
            # Fallback to anything with "EMAIL" that ISN'T a hash
            potential_cols = [c for c in df.columns if 'EMAIL' in c and 'SHA' not in c and 'HEM' not in c]
            target_email_col = potential_cols[0] if potential_cols else df.columns[0]
        
        st.sidebar.write(f"🔍 Identity Graph matching on AWS column: **{target_email_col}**")
        
        df = df.rename(columns={target_email_col: 'Email'})
        df['Email'] = df['Email'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.split(',')
        df = df.explode('Email').reset_index(drop=True)
        df['Email'] = df['Email'].str.strip()
        
        return df.drop_duplicates(subset=['Email'], keep='first').reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 AWS Matcher Error: {e}"); st.stop()

# ================ 3. STATE 1: ONBOARDING =================
if st.session_state.app_state == "onboarding":
    st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem;'>{PITCH_COMPANY_NAME}</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded_file:
        df_orders = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
        df_orders = df_orders.rename(columns={'Name': 'Order ID', 'Created at': 'Date'})
        
        with st.spinner("Executing Identity Resolution..."):
            df_master = load_master_graph()
            
            # Clean order emails identically to the master list
            df_orders['Email'] = df_orders['Email'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.strip()
            
            # Diagnostic count
            unique_orders = df_orders['Email'].nunique()
            st.sidebar.write(f"📊 Unique Emails Uploaded: {unique_orders}")
            
            df_joined = pd.merge(df_orders, df_master, on='Email', how='inner').reset_index(drop=True)
            
            if not df_joined.empty:
                st.session_state.df_icp = df_joined
                st.session_state.app_state = "dashboard"
                st.rerun()
            else:
                st.error("⚠️ Zero Matches. Check sidebar diagnostics for column names.")

# ================ 4. STATE 2: DASHBOARD =================
elif st.session_state.app_state == "dashboard":
    st.markdown(f"## 🧬 Identity Match Result")
    if st.button("← New Analysis", type="secondary"): st.session_state.app_state = "onboarding"; st.rerun()
    
    df = st.session_state.df_icp
    df['Total'] = pd.to_numeric(df['Total'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)

    m1, m2 = st.columns(2)
    m1.metric("Resolved Profiles", f"{df['Order ID'].nunique():,.0f}")
    m2.metric("Attributed Sales", f"${df['Total'].sum():,.2f}")
    st.markdown("<hr>", unsafe_allow_html=True)

    configs = [
        ("Gender", "gender"), ("Marital Status", "marital_status"), ("Credit Rating", "credit_rating"), 
        ("Household Income", "income"), ("Net Worth", "net_worth"), ("Geographic Region", "region"), ("Age Range", "age")
    ]

    for label, col in configs:
        if col in df.columns:
            df_plot = df[~df[col].astype(str).str.lower().isin(['u', 'unknown', 'nan', 'none', '', 'other'])]
            grp = df_plot.groupby(col).agg(Buyers=('Order ID', 'nunique'), Revenue=('Total', 'sum')).reset_index()
            
            if not grp.empty:
                st.markdown(f"<h2 style='text-align: center; margin-bottom: 2rem;'>{label} Distribution</h2>", unsafe_allow_html=True)
                chart = alt.Chart(grp).mark_arc(innerRadius=85, stroke="#fff").encode(
                    theta="Revenue:Q", color=alt.Color(f"{col}:N", scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(title=label, orient="right", labelFontSize=14)),
                    tooltip=[alt.Tooltip(f'{col}:N', title=label), alt.Tooltip('Revenue:Q', format='$,.0f')]
                ).properties(width=700, height=450)
                st.altair_chart(chart, use_container_width=False)
                grp['% Share'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col: label})
                render_premium_table(grp.style.format({'Buyers': '{:,.0f}', 'Revenue': '${:,.2f}', '% Share': '{:.1f}%', 'AOV': '${:,.2f}'}).background_gradient(subset=['% Share'], cmap=custom_light_green))
