import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import os
import altair as alt

# ================ 0. PITCH CONFIGURATION =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#0A2540" 

# 🚨 THE DATA MAPPER 🚨
# Updated to match your exact AWS column names
AWS_COLUMN_MAPPER = {
    "GENDER": "gender",
    "AGE_RANGE": "age",
    "INCOME_RANGE": "income",
    "PERSONAL_STATE": "region",
    "NET_WORTH": "net_worth",
    "CHILDREN": "children",
    "MARRIED": "marital_status",
    "HOMEOWNER": "homeowner",
    "SKIPTRACE_CREDIT_RATING": "credit_rating"
}
# =========================================================

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Customer DNA", page_icon="🧬", layout="centered", initial_sidebar_state="collapsed")

if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"
if "df_icp" not in st.session_state: st.session_state.df_icp = None

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 600 !important; }}
            
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; border-radius: 8px; font-weight: 800; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; border-radius: 8px; }}
            
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px 24px; box-shadow: 0 4px 10px rgba(45, 36, 33, 0.04); }}
            [data-testid="stMetricLabel"] {{ color: {primary_color} !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.85rem; }}
            
            .premium-table-container {{ width: 100%; overflow-x: auto; border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; margin-bottom: 3rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; padding: 12px 14px !important; border-bottom: 2px solid #D5C6B3 !important; text-transform: uppercase !important; font-size: 0.75rem !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 10px 14px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.9rem !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

def render_premium_table(styler_obj):
    try: styler_obj = styler_obj.hide(axis="index")
    except AttributeError: styler_obj = styler_obj.hide_index() 
    html = styler_obj.to_html()
    st.markdown(f'<div class="premium-table-container">{html}</div>', unsafe_allow_html=True)

configs = [
    ("Gender", "gender"), 
    ("Age Range", "age"), 
    ("Household Income", "income"), 
    ("State/Region", "region"), 
    ("Net Worth", "net_worth"), 
    ("Children Presence", "children"), 
    ("Marital Status", "marital_status"), 
    ("Homeowner Status", "homeowner"), 
    ("Credit Rating", "credit_rating")
]

# ================ 2. LIVE AWS CONNECTION =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    YOUR_AWS_REGION = "us-east-2" 
    aws_keys = {
        "key": st.secrets["aws"]["access_key"],
        "secret": st.secrets["aws"]["secret_key"],
        "client_kwargs": {"region_name": YOUR_AWS_REGION} 
    }
    s3_file_path = "s3://leadnav-demo-data/master_data.csv"
    
    try:
        df_master = pd.read_csv(s3_file_path, storage_options=aws_keys, low_memory=False)
        
        # Mapping
        rename_dict = {k.lower(): v for k, v in AWS_COLUMN_MAPPER.items()}
        current_cols_lower = {c.lower(): c for c in df_master.columns}
        for map_key, map_val in rename_dict.items():
            if map_key in current_cols_lower:
                df_master = df_master.rename(columns={current_cols_lower[map_key]: map_val})
        
        # Explode Emails
        email_cols = [col for col in df_master.columns if 'email' in col.lower()]
        if email_cols:
            df_master = df_master.rename(columns={email_cols[0]: 'Email'})
            df_master['Email'] = df_master['Email'].astype(str).str.lower().str.split(',').explode().str.strip()
            df_master = df_master.drop_duplicates(subset=['Email'], keep='first')
            
        return df_master
    except Exception as e:
        st.error(f"🚨 **AWS Connection Error:** {str(e)}")
        st.stop()

# ================ 3. STATE 1: ONBOARDING SCREEN =================
if st.session_state.app_state == "onboarding":
    col_logo1, col_logo2, col_logo3 = st.columns([1, 1, 1])
    with col_logo2:
        if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
            
    st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; color: {PITCH_BRAND_COLOR} !important;'>{PITCH_COMPANY_NAME}</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 1.5rem; color: #444;'>Identity Resolution & Customer DNA</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])

    if uploaded_file is not None:
        df_orders = pd.read_csv(uploaded_file)
        shopify_map = {'Name': 'Order ID', 'Created at': 'Date'}
        df_orders = df_orders.rename(columns=shopify_map)
        
        if "Email" not in df_orders.columns:
            st.error("⚠️ CSV missing 'Email' column.")
        else:
            with st.spinner("Accessing Identity Graph..."):
                df_master_aws = load_master_graph()
                df_orders['Total'] = pd.to_numeric(df_orders['Total'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)
                df_orders['Email'] = df_orders['Email'].astype(str).str.lower().str.strip()
                
                df_joined = pd.merge(df_orders, df_master_aws, on='Email', how='inner')
                
                if df_joined.empty:
                    st.error("⚠️ Zero matches found.")
                else:
                    st.session_state.df_icp = df_joined
                    st.session_state.app_state = "dashboard"
                    st.rerun()

# ================ 4. STATE 2: LIVE DASHBOARD =================
elif st.session_state.app_state == "dashboard":
    st.markdown(f"<h1 style='color: {PITCH_BRAND_COLOR} !important;'>🧬 Customer DNA Analysis</h1>", unsafe_allow_html=True)
    
    if st.button("← Back to Upload", type="secondary"):
        st.session_state.app_state = "onboarding"
        st.rerun()
        
    df_joined = st.session_state.df_icp
    
    # KPIs
    total_buyers = df_joined['Order ID'].nunique()
    total_rev = df_joined['Total'].sum()
    m1, m2 = st.columns(2)
    m1.metric("Resolved Buyers", f"{total_buyers:,.0f}")
    m2.metric("Attributed Revenue", f"${total_rev:,.2f}")
    st.markdown("<hr>", unsafe_allow_html=True)

    # LOOP THROUGH VARIABLES
    for label, col_name in configs:
        if col_name in df_joined.columns:
            # FILTER: Remove "U", "Unknown", "nan"
            df_filtered = df_joined[~df_joined[col_name].astype(str).str.lower().isin(['u', 'unknown', 'nan', 'none', 'null'])]
            
            grp = df_filtered.groupby(col_name).agg(Buyers=('Order ID', 'nunique'), Revenue=('Total', 'sum')).reset_index()
            
            if not grp.empty:
                st.markdown(f"### {label}")
                
                # --- PIE CHART ---
                pie_chart = alt.Chart(grp).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Revenue", type="quantitative"),
                    color=alt.Color(field=col_name, type="nominal", scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(title=label)),
                    tooltip=[col_name, alt.Tooltip('Revenue', format='$,.0f')]
                ).properties(height=400)
                
                st.altair_chart(pie_chart, use_container_width=True)
                
                # --- TABLE ---
                grp['% Revenue'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col_name: label})
                
                format_dict = {'Buyers': '{:,.0f}', 'Revenue': '${:,.2f}', '% Revenue': '{:.1f}%', 'AOV': '${:,.2f}'}
                styler = grp.style.format(format_dict).background_gradient(subset=['% Revenue'], cmap=custom_light_green)
                render_premium_table(styler)
