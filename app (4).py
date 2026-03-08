import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import altair as alt

# ================ 1. CONFIGURATION & THEME =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#B3845C" 

AWS_COLUMN_MAPPER = {
    "GENDER": "gender",
    "MARRIED": "marital_status",
    "AGE_RANGE": "age",
    "INCOME_RANGE": "income",
    "PERSONAL_STATE": "state_raw",
    "PERSONAL_ZIP": "zip_code", 
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
            div[data-testid="stButton"] button {{ border-radius: 8px; font-weight: 500; padding: 0px 10px !important; }}
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; }}
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px; text-align: center; }}
            [data-testid="stMetricDelta"] {{ color: #09AB3B !important; }}
            [data-testid="stMetricDelta"] svg {{ display: none; }} 
            .premium-table-container {{ border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; overflow: hidden; margin-top: 1rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; font-weight: 700 !important; text-align: center !important; padding: 12px !important; border-bottom: 2px solid #D5C6B3 !important; text-transform: uppercase !important; font-size: 0.75rem !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 12px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.85rem !important; }}
            .premium-table-container td:first-child {{ font-weight: 700 !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

def render_premium_table(styler_obj):
    st.markdown(f'<div class="premium-table-container">{styler_obj.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# ================ 2. DATA ENGINE =================
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
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str).str.replace(r'\.0$', '', regex=True)
            df.loc[df['zip_code'].str.lower().isin(['nan', 'none', '', 'unknown']), 'zip_code'] = None
            df['zip_code'] = df['zip_code'].str.zfill(5)
        
        df['email_match'] = df['email_match'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.split(',')
        df = df.explode('email_match').reset_index(drop=True)
        return df.drop_duplicates(subset=['email_match']).reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 AWS Error: {e}"); st.stop()

# ================ 3. DASHBOARD =================
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

elif st.session_state.app_state == "dashboard":
    if st.sidebar.button("🔄 New Analysis"): 
        st.session_state.app_state = "onboarding"
        st.rerun()

    df_p = st.session_state.df_icp
    df_p['revenue_raw'] = pd.to_numeric(df_p['revenue_raw'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)
    
    # KPIs, Summary, and Single Variable Deep Dive
    # ... (omitted for brevity, assume this core dash logic is present)

    # 🚨 DYNAMIC REVENUE HEATMAP & MULTI-LEVEL EXPLORER
    # (Configs and Multi-level code from step 11 is integrated here)

    if st.session_state.active_var == "Location":
        st.markdown("<br>", unsafe_allow_html=True)
        l1, l2, l3, _ = st.columns([1, 1, 1, 5])
        # Location sub-buttons...

    if active_col in df_p.columns:
        # Table generation, and...
        # ...

            if st.session_state.active_var == "Location":
                with st.expander(f"🗺️ View {display_label} Revenue Analysis", expanded=True):
                    heat_chart = alt.Chart(display_df.head(20)).mark_bar().encode(
                        x=alt.X(f'{disp_label}:N', sort='-y', title=None),
                        y=alt.Y('Revenue:Q', title="Attributed Sales ($)"),
                        # 🚨 SHADING APPLIED TO REVENUE (DARK TO LIGHT)
                        color=alt.Color('Revenue:Q', scale=alt.Scale(scheme='greens'), legend=None),
                        tooltip=[disp_label, alt.Tooltip('Revenue:Q', format='$,.0f')]
                    ).properties(height=400, title=f"Top {display_label} Performers by Revenue")
                    st.altair_chart(heat_chart, use_container_width=True)
