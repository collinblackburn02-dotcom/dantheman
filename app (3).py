import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import altair as alt

# ================ 1. CONFIGURATION =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#0A2540" 
DEMO_PASSWORD = "leadnavai"

AWS_COLUMN_MAPPER = {
    "GENDER": "gender",
    "MARRIED": "marital_status",
    "AGE_RANGE": "age",
    "INCOME_RANGE": "income",
    "PERSONAL_STATE": "state_raw",
    "NET_WORTH": "net_worth",
    "SKIPTRACE_CREDIT_RATING": "credit_rating"
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

# BUCKETING
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

# ================ 2. AWS LOADER =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    aws_keys = {"key": st.secrets["aws"]["access_key"], "secret": st.secrets["aws"]["secret_key"], "client_kwargs": {"region_name": "us-east-2"}}
    files = ["master_data.csv", "visitor_data_2.csv"] 
    dataframes = []
    
    try:
        for f in files:
            path = f"s3://leadnav-demo-data/{f}"
            temp_df = pd.read_csv(path, storage_options=aws_keys, low_memory=False, encoding='latin1', on_bad_lines='skip')
            temp_df.columns = [c.upper().strip() for c in temp_df.columns]
            
            # Use PERSONAL_EMAILS specifically
            if 'PERSONAL_EMAILS' in temp_df.columns:
                temp_df = temp_df.rename(columns={'PERSONAL_EMAILS': 'email_match'})
            else:
                email_fallback = next((c for c in temp_df.columns if 'EMAIL' in c and 'SHA' not in c), temp_df.columns[0])
                temp_df = temp_df.rename(columns={email_fallback: 'email_match'})
            
            dataframes.append(temp_df)
            
        df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
        df = df.rename(columns=AWS_COLUMN_MAPPER)
        df.columns = [c.lower() for c in df.columns]
        
        if 'state_raw' in df.columns: 
            df['region'] = df['state_raw'].str.strip().str.upper().map(STATE_TO_REGION).fillna('Other')
        if 'income' in df.columns: df['income'] = df['income'].apply(bucket_income)
        if 'net_worth' in df.columns: df['net_worth'] = df['net_worth'].apply(bucket_nw)
        if 'credit_rating' in df.columns: df['credit_rating'] = df['credit_rating'].apply(bucket_credit)
        
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')
        if 'marital_status' in df.columns:
            df['marital_status'] = df['marital_status'].map({'Y': 'Married', 'N': 'Single'}).fillna('Unknown')
        
        df['email_match'] = df['email_match'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.split(',')
        df = df.explode('email_match').reset_index(drop=True)
        df['email_match'] = df['email_match'].str.strip()
        
        return df.drop_duplicates(subset=['email_match'], keep='first').reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 AWS Matcher Error: {e}"); st.stop()

# ================ 3. ONBOARDING =================
if st.session_state.app_state == "onboarding":
    st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem;'>{PITCH_COMPANY_NAME}</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded_file:
        df_orders = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
        df_orders = df_orders.rename(columns={'Email': 'email_match', 'Name': 'Order ID', 'Total': 'revenue_raw'})
        
        with st.spinner("Executing LeadNavigator Identity Resolution..."):
            df_master = load_master_graph()
            df_orders['email_match'] = df_orders['email_match'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.strip()
            
            # Match ENTIRE file first to maximize matches
            df_joined = pd.merge(df_orders, df_master, on='email_match', how='inner').reset_index(drop=True)
            
            if not df_joined.empty:
                st.session_state.df_icp = df_joined
                st.session_state.app_state = "dashboard"
                st.rerun()
            else:
                st.error("⚠️ Zero matches found.")

# ================ 4. DASHBOARD =================
elif st.session_state.app_state == "dashboard":
    with st.sidebar:
        st.title("🔒 Security")
        pwd = st.text_input("Enter Dashboard Password", type="password")
        is_unlocked = (pwd == DEMO_PASSWORD)
        if is_unlocked:
            st.success("Full Access Granted")
        else:
            st.warning("Preview Mode: First 100 Matches")

    st.markdown(f"## 🧬 Identity Match Result {'(Unlocked)' if is_unlocked else '(Restricted)'}")
    if st.button("← New Analysis", type="secondary"): st.session_state.app_state = "onboarding"; st.rerun()
    
    # Clip only after matching is done
    full_df = st.session_state.df_icp
    df = full_df.head(100).copy() if not is_unlocked else full_df.copy()

    df['revenue_raw'] = pd.to_numeric(df['revenue_raw'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)

    m1, m2 = st.columns(2)
    m1.metric("Resolved Profiles", f"{df['Order ID'].nunique():,.0f}")
    with m2: st.metric("Attributed Sales", f"${df['revenue_raw'].sum():,.2f}")
    st.markdown("<hr>", unsafe_allow_html=True)

    configs = [
        ("Gender", "gender"), ("Marital Status", "marital_status"), ("Credit Rating", "credit_rating"), 
        ("Household Income", "income"), ("Net Worth", "net_worth"), ("Geographic Region", "region"), ("Age Range", "age")
    ]

    for label, col in configs:
        if col in df.columns:
            chart_data = df.copy()
            chart_data = chart_data[~chart_data[col].astype(str).str.lower().isin(['u', 'nan', 'none', '', 'unknown', 'other'])]
            
            if not chart_data.empty:
                grp = chart_data.groupby(col).agg(Buyers=('Order ID', 'nunique'), Revenue=('revenue_raw', 'sum')).reset_index()
                st.markdown(f"<h2 style='text-align: center; margin-bottom: 2rem;'>{label} Distribution</h2>", unsafe_allow_html=True)
                
                chart = alt.Chart(grp).mark_arc(innerRadius=85, stroke="#fff").encode(
                    theta="Revenue:Q", 
                    color=alt.Color(f"{col}:N", scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(title=label, orient="right", labelFontSize=14)),
                    tooltip=[alt.Tooltip(f'{col}:N', title=label), alt.Tooltip('Revenue:Q', format='$,.0f')]
                ).properties(width=700, height=450)
                
                st.altair_chart(chart, use_container_width=False)
                
                grp['% Share'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col: label})
                
                render_premium_table(grp.style.format({'Buyers': '{:,.0f}', 'Revenue': '${:,.2f}', '% Share': '{:.1f}%', 'AOV': '${:,.2f}'}).background_gradient(subset=['% Share'], cmap=custom_light_green))
