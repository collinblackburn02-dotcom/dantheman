import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import os

# ================ 0. PITCH CONFIGURATION =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#0A2540" 
# =========================================================

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Customer DNA", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

# --- App State Management ---
if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"
if "df_icp" not in st.session_state: st.session_state.df_icp = None

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 600 !important; }}
            p, span, label {{ color: #2D2421 !important; }}
            
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; border-radius: 8px; font-weight: 800; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; border-radius: 8px; }}
            
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px 24px; box-shadow: 0 4px 10px rgba(45, 36, 33, 0.04); }}
            [data-testid="stMetricLabel"] {{ color: {primary_color} !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.85rem; }}
            [data-testid="stMetricValue"] {{ color: #2D2421 !important; font-weight: 700; font-size: 2.2rem; }}
            
            .premium-table-container {{ width: 100%; overflow-x: auto; border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; margin-bottom: 2rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; font-family: 'Outfit', sans-serif !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; font-weight: 700 !important; text-align: center !important; padding: 12px 14px !important; border-bottom: 2px solid #D5C6B3 !important; text-transform: uppercase !important; font-size: 0.70rem !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 10px 14px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.85rem !important; }}
            .premium-table-container td:first-child {{ font-weight: 700 !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

def render_premium_table(styler_obj):
    try: styler_obj = styler_obj.hide(axis="index")
    except AttributeError: styler_obj = styler_obj.hide_index() 
    html = styler_obj.to_html()
    st.markdown(f'<div class="premium-table-container">{html}</div>', unsafe_allow_html=True)

configs = [("Gender", "gender"), ("Age", "age"), ("Income", "income"), ("Region", "region"), ("Net Worth", "net_worth"), ("Children", "children"), ("Marital Status", "marital_status"), ("Homeowner", "homeowner"), ("Credit Rating", "credit_rating")]

# ================ 2. LIVE AWS CONNECTION =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    """Reads your live master list directly from AWS S3."""
    
    # Explicitly pointing to Ohio!
    YOUR_AWS_REGION = "us-east-2" 
    
    aws_keys = {
        "key": st.secrets["aws"]["access_key"],
        "secret": st.secrets["aws"]["secret_key"],
        "client_kwargs": {"region_name": YOUR_AWS_REGION} 
    }
    
    s3_file_path = "s3://leadnav-demo-data/master_data.csv"
    
    try:
        # Read directly from AWS into memory
        df_master = pd.read_csv(s3_file_path, storage_options=aws_keys, low_memory=False)
        
        if 'Email' in df_master.columns:
            df_master['Email'] = df_master['Email'].astype(str).str.lower().str.strip()
            
        return df_master
        
    except Exception as e:
        # If anything goes wrong, this will print the REAL AWS error on your dashboard!
        st.error(f"🚨 **AWS Connection Error:** {str(e)}")
        st.stop()

# ================ 3. STATE 1: ONBOARDING SCREEN =================
if st.session_state.app_state == "onboarding":
    
    col_logo1, col_logo2, col_logo3 = st.columns([2, 1, 2])
    with col_logo2:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
            
    st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; color: {PITCH_BRAND_COLOR} !important; margin-top: 10px;'>{PITCH_COMPANY_NAME}</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 1.8rem; margin-top: -10px; color: #444;'>Identity Resolution Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 3rem; font-size: 1.1rem;'>Upload your raw orders. We will securely match them against our identity graph to reveal your Customer DNA.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Process Historical Orders")
        st.info("📦 **Format Required:** `Date`, `Email`, `Order ID`, `Total` (Shopify exports work automatically!)")
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            df_orders = pd.read_csv(uploaded_file)
            
            # Shopify Auto-Mapper
            shopify_map = {'Name': 'Order ID', 'Created at': 'Date'}
            df_orders = df_orders.rename(columns=shopify_map)
            
            required_cols = ["Date", "Email", "Order ID", "Total"]
            missing_cols = [col for col in required_cols if col not in df_orders.columns]
            
            if missing_cols:
                st.error(f"⚠️ Your CSV is missing: **{', '.join(missing_cols)}**")
            else:
                with st.spinner(f"Pulling {PITCH_COMPANY_NAME} Identity Graph from AWS S3..."):
                    df_master_aws = load_master_graph()
                    
                    with st.spinner("Resolving Identities and Calculating Match Rate..."):
                        df_orders['Total'] = df_orders['Total'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        df_orders['Total'] = pd.to_numeric(df_orders['Total'], errors='coerce').fillna(0)
                        
                        # FIX 1: Use 'Order ID' instead of 'Order_ID'
                        df_orders = df_orders.drop_duplicates(subset=['Order ID'], keep='first')
                        
                        # === THE LEAD MAGNET GATEKEEPER ===
                        if len(df_orders) > 1000:
                            st.session_state.truncated = True
                            df_orders = df_orders.head(1000)
                        else:
                            st.session_state.truncated = False
                        
                        df_orders['Email'] = df_orders['Email'].astype(str).str.lower().str.strip()
                        
                        # === REAL AWS MATCH LOGIC ===
                        df_joined = pd.merge(df_orders, df_master_aws, on='Email', how='inner')
                        
                        if df_joined.empty:
                            st.error("⚠️ Zero matches found. Make sure the 'Email' column in your AWS CSV exactly matches the uploaded orders.")
                        else:
                            # Standardize nulls for clean charts
                            df_joined = df_joined.fillna('Unknown').replace(["", "nan", "NaN", "None", "null", "NULL", "<NA>"], "Unknown")
                            st.session_state.df_icp = df_joined
                            st.session_state.app_state = "dashboard"
                            st.rerun()

# ================ 4. STATE 2: LIVE DASHBOARD =================
elif st.session_state.app_state == "dashboard":
    
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown(f"<h1 style='color: {PITCH_BRAND_COLOR} !important; margin-bottom: 0px;'>🧬 Customer DNA Match</h1>", unsafe_allow_html=True)
    with col_head2:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=150)
    
    # Gatekeeper Alert
    if st.session_state.get('truncated', False):
        st.warning(f"⚠️ **Lead Magnet Preview:** Your file exceeded 1,000 rows. We resolved identities for the first 1,000 orders to demonstrate the {PITCH_COMPANY_NAME} Engine. Connect with our team to unlock your full historical database.")
    else:
        st.markdown(f"<p style='color: #666; font-size: 1.1rem; margin-top: -5px; margin-bottom: 30px;'>Successfully resolved identities via the {PITCH_COMPANY_NAME} Graph.</p>", unsafe_allow_html=True)
    
    if st.button("← Upload New File", type="secondary"):
        st.session_state.app_state = "onboarding"
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    df_joined = st.session_state.df_icp
    
    # FIX 2: Use 'Order ID' instead of 'Order_ID'
    total_buyers = df_joined['Order ID'].nunique()
    total_rev = df_joined['Total'].sum()
    overall_aov = total_rev / total_buyers if total_buyers > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Resolved Buyers", f"{total_buyers:,.0f}")
    m2.metric("Total Attributed Revenue", f"${total_rev:,.2f}")
    m3.metric("Overall Average Order Value", f"${overall_aov:,.2f}")
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    dash_col1, dash_col2 = st.columns(2)
    for index, (label, col_name) in enumerate(configs):
        if col_name in df_joined.columns:
            # FIX 3: Use 'Order ID' instead of 'Order_ID'
            grp = df_joined[df_joined[col_name] != 'Unknown'].groupby(col_name).agg(Buyers=('Order ID', 'nunique'), Revenue=('Total', 'sum')).reset_index()
            
            if not grp.empty:
                grp['% of Buyers'] = (grp['Buyers'] / grp['Buyers'].sum()) * 100
                grp['% of Revenue'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col_name: label})
                
                format_dict = {'Buyers': '{:,.0f}', '% of Buyers': '{:.1f}%', 'Revenue': '${:,.2f}', '% of Revenue': '{:.1f}%', 'AOV': '${:,.2f}'}
                styler = grp.style.format(format_dict).background_gradient(subset=['% of Revenue', '% of Buyers'], cmap=custom_light_green)
                
                with dash_col1 if index % 2 == 0 else dash_col2:
                    st.subheader(f"{label} Identity Distribution")
                    render_premium_table(styler)
