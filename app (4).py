import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors

# ================ 1. CONFIGURATION & THEME =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#B3845C" # Premium Tan
DEMO_PASSWORD = "leadnavai"

# Hardcoded AWS headers
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

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Audience Engine", page_icon="🧬", layout="wide")

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 600 !important; }}
            
            /* Selection Bar Buttons */
            div[data-testid="stButton"] button {{ border-radius: 8px; font-weight: 500; padding: 0px 10px !important; }}
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; }}
            
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px; }}
            
            /* Premium Table Formatting */
            .premium-table-container {{ border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; overflow: hidden; margin-top: 1rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; font-weight: 700 !important; text-align: center !important; padding: 12px !important; text-transform: uppercase !important; font-size: 0.75rem !important; border-bottom: 2px solid #D5C6B3 !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 12px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.85rem !important; }}
            .premium-table-container td:first-child {{ font-weight: 700 !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

def render_premium_table(styler_obj):
    st.markdown(f'<div class="premium-table-container">{styler_obj.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# ================ 2. AWS DATA ENGINE =================
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
            e_col = 'PERSONAL_EMAILS' if 'PERSONAL_EMAILS' in temp_df.columns else temp_df.columns[0]
            temp_df = temp_df.rename(columns={e_col: 'email_match'})
            dataframes.append(temp_df)
        df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
        df = df.rename(columns=AWS_COLUMN_MAPPER)
        df.columns = [c.lower() for c in df.columns]
        if 'state_raw' in df.columns: df['region'] = df['state_raw'].str.strip().str.upper().map(STATE_TO_REGION).fillna('Unknown')
        if 'gender' in df.columns: df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')
        if 'marital_status' in df.columns: df['marital_status'] = df['marital_status'].map({'Y': 'Married', 'N': 'Single'}).fillna('Unknown')
        df['email_match'] = df['email_match'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.split(',')
        df = df.explode('email_match').reset_index(drop=True)
        return df.drop_duplicates(subset=['email_match']).reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 AWS Matcher Error: {e}"); st.stop()

# ================ 3. ONBOARDING =================
if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"
if st.session_state.app_state == "onboarding":
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-top: 50px;'>🎯 Audience Engine</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])
        if uploaded_file:
            df_orders = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
            df_orders = df_orders.rename(columns={'Email': 'email_match', 'Name': 'Order ID', 'Total': 'revenue_raw'})
            with st.spinner("Executing LeadNavigator Resolution..."):
                df_master = load_master_graph()
                df_orders['email_match'] = df_orders['email_match'].astype(str).str.lower().str.strip()
                df_joined = pd.merge(df_orders, df_master, on='email_match', how='inner').reset_index(drop=True)
                if not df_joined.empty:
                    st.session_state.df_icp = df_joined
                    st.session_state.app_state = "dashboard"
                    st.rerun()

# ================ 4. DASHBOARD (DEEP DIVE) =================
elif st.session_state.app_state == "dashboard":
    with st.sidebar:
        st.title("🔒 Security")
        pwd = st.text_input("Password", type="password")
        is_unlocked = (pwd == DEMO_PASSWORD)
        if st.button("🔄 New Analysis"): st.session_state.app_state = "onboarding"; st.rerun()

    st.markdown("### 🔍 Single Variable Deep Dive")
    
    configs = [("Gender", "gender"), ("Age", "age"), ("Region", "region"), ("Marital Status", "marital_status"), ("Credit Rating", "credit_rating")]
    if "active_var" not in st.session_state: st.session_state.active_var = "Gender"
    
    # 🚨 HORIZONTAL SELECTOR 🚨
    var_cols = st.columns(len(configs))
    for i, (label, col_name) in enumerate(configs):
        if var_cols[i].button(label, key=f"btn_{label}", type="primary" if st.session_state.active_var == label else "secondary", use_container_width=True):
            st.session_state.active_var = label
            st.rerun()

    active_label = st.session_state.active_var
    active_col = dict(configs)[active_label]

    # Data Clipping
    full_p = st.session_state.df_icp
    if not is_unlocked:
        top_100_ids = full_p['Order ID'].unique()[:100]
        df_p = full_p[full_p['Order ID'].isin(top_100_ids)].copy()
    else:
        df_p = full_p.copy()

    df_p['revenue_raw'] = pd.to_numeric(df_p['revenue_raw'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)
    
    m1, m2 = st.columns(2)
    m1.metric("Resolved Profiles", f"{df_p['Order ID'].nunique():,.0f}")
    m2.metric("Attributed Sales", f"${df_p['revenue_raw'].sum():,.2f}")

    # 🚨 UPDATED MATH: Revenue per Purchaser (AOV) only
    df_p_grp = df_p.groupby(active_col).agg(
        Purchasers=('Order ID', 'nunique'), 
        Revenue=('revenue_raw', 'sum')
    ).reset_index()
    
    df_p_grp = df_p_grp[~df_p_grp[active_col].astype(str).str.lower().isin(['unknown', 'nan', 'other', 'u'])]
    
    if not df_p_grp.empty:
        df_p_grp['% of Buyers'] = (df_p_grp['Purchasers'] / df_p_grp['Purchasers'].sum()) * 100
        df_p_grp['Rev / Purchaser'] = (df_p_grp['Revenue'] / df_p_grp['Purchasers'])
        
        display_df = df_p_grp.rename(columns={active_col: active_label.upper()}).sort_values('Revenue', ascending=False)
        
        # Premium Table Styles
        styler = display_df.style.format({
            'Purchasers': '{:,.0f}', 
            'Revenue': '${:,.2f}', 
            '% of Buyers': '{:.1f}%',
            'Rev / Purchaser': '${:,.2f}'
        }).background_gradient(subset=['Rev / Purchaser', '% of Buyers'], cmap=custom_light_green)
        
        render_premium_table(styler)
