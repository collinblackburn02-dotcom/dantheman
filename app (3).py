import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import os
import altair as alt

# ================ ZLIB & PITCH CONFIGURATION =================
# If your app runs locally but crashes on Streamlit Cloud, you need to add zlib.
# It is critical for pandas to read compressed S3 data.
# 🚨 Add 'libz-dev' to a new file named 'packages.txt' in your repo's root folder!

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
            [data-testid="stMetricLabel"] {{ color: {primary_color} !important; }}
            [data-testid="stMetricValue"] {{ color: #2D2421 !important; font-size: 2.5rem; }}
            .premium-table-container {{ border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; margin-bottom: 5rem; width: fit-content; max-width: 100%; }}
            .premium-table-container table {{ width: auto !important; border-collapse: collapse !important; table-layout: fixed; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; text-transform: uppercase; font-size: 0.75rem; text-align: center !important; white-space: nowrap; }}
            .premium-table-container td {{ text-align: center !important; padding: 10px 14px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.9rem !important; }}
            [data-testid="column"] {{ text-align: left; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#6EAB6E"])

def render_premium_table(styler_obj):
    st.markdown(f'<div class="premium-table-container">{styler_obj.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# 🚨 UPDATED SIMPLIFIED BUCKETING HELPERS
def bucket_income(val):
    v = str(val).lower()
    if any(x in v for x in ['250', '500']): return "High"
    if any(x in v for x in ['125', '150', '175', '200']): return "Medium-High"
    if any(x in v for x in ['50', '75', '100']): return "Medium"
    return "Low"

def bucket_nw(val):
    v = str(val).lower()
    if any(x in v for x in ['1,000', '2,000', '5,000']): return "High"
    if any(x in v for x in ['250', '500']): return "Medium-High"
    if any(x in v for x in ['50', '100']): return "Medium"
    return "Low"

def bucket_credit(val):
    v = str(val).upper().strip()
    # Range is ABC -> Elite/Prime. FG -> Low/Developing.
    # Grouping into 3 logical simplified categories matching 3 original number logical categories.
    if v in ['A', 'B', 'C']: return "High" # Elite/Prime
    if v in ['D', 'E']: return "Medium" # Standard
    if v in ['F', 'G']: return "Low" # Developing
    return "U" # Unknown

# ================ 2. LIVE AWS CONNECTION =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    aws_keys = {"key": st.secrets["aws"]["access_key"], "secret": st.secrets["aws"]["secret_key"], "client_kwargs": {"region_name": "us-east-2"}}
    try:
        # Check context again - zlib libz-dev required for cloud.
        df = pd.read_csv("s3://leadnav-demo-data/master_data.csv", storage_options=aws_keys, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        
        df = df.reset_index(drop=True)
        
        rename_dict = {k.lower(): v for k, v in AWS_COLUMN_MAPPER.items()}
        df = df.rename(columns=rename_dict)
        
        # B2C Transformation with logic. Applying mapping logic to raw columns.
        if 'state_raw' in df.columns: df['region'] = df['state_raw'].str.strip().str.upper().map(STATE_TO_REGION)
        if 'income_raw' in df.columns: df['income'] = df['income_raw'].apply(bucket_income)
        if 'net_worth_raw' in df.columns: df['net_worth'] = df['net_worth_raw'].apply(bucket_nw)
        if 'credit_raw' in df.columns: df['credit_rating'] = df['credit_raw'].apply(bucket_credit)
        
        # Email Explosion logic
        email_col = next((c for c in df.columns if 'email' in c.lower()), 'Email')
        df = df.rename(columns={email_col: 'Email'})
        df['Email'] = df['Email'].astype(str).str.lower().str.split(',')
        
        df = df.explode('Email')
        df['Email'] = df['Email'].str.strip()
        
        # FINAL Deduplication Shield
        return df.drop_duplicates(subset=['Email'], keep='first').reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 AWS Error: {e}"); st.stop()

# ================ 3. DASHBOARD LOGIC =================
if st.session_state.app_state == "onboarding":
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
            
    st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; color: {PITCH_BRAND_COLOR};'>{PITCH_COMPANY_NAME}</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 1.5rem; color: #444; margin-top: -15px;'>Identity Resolution Engine</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Unlock Customer DNA. Upload your sales/order CSV. We securely match emails against our identity graph to reveal demographic and financial profiles.</p>", unsafe_allow_html=True)
        st.markdown("### Process Historical Orders")
        uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            df_orders = pd.read_csv(uploaded_file)
            shopify_map = {'Name': 'Order ID', 'Created at': 'Date'}
            df_orders = df_orders.rename(columns=shopify_map)
            
            if "Email" not in df_orders.columns:
                st.error("⚠️ CSV missing 'Email' column.")
            else:
                with st.spinner("Connecting to Identity Graph..."):
                    df_master_aws = load_master_graph()
                    # Shopify mapper
                    df_orders['Email'] = df_orders['Email'].astype(str).str.lower().str.strip()
                    
                    # Merge logic
                    df_joined = pd.merge(df_orders, df_master_aws, on='Email', how='inner').reset_index(drop=True)
                    
                    if not df_joined.empty:
                        st.session_state.df_icp = df_joined
                        st.session_state.app_state = "dashboard"
                        st.rerun()

elif st.session_state.app_state == "dashboard":
    st.markdown(f"<h1 style='color: {PITCH_BRAND_COLOR};'>🧬 Identity Match Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 1.1rem; margin-top: -15px; margin-bottom: 20px;'>Successfully resolved identities via the LeadNavigator Identity Graph.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← New Analysis", type="secondary"): st.session_state.app_state = "onboarding"; st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    df = st.session_state.df_icp
    # Format and clean total before KPI calc
    df['Total'] = pd.to_numeric(df['Total'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)

    col_metric1, col_metric2 = st.columns(2)
    with col_metric1: st.metric("Resolved Profiles", f"{df['Order ID'].nunique():,.0f}")
    with col_metric2: st.metric("Attributed Sales", f"${df['Total'].sum():,.2f}")
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # LOOP THROUGH B2C VARIABLES FOR VISUALS
    display_configs = [
        ("Credit Rating Score", "credit_rating"), ("Household Income", "income"), ("Net Worth Segment", "net_worth"), 
        ("Geographic Region", "region"), ("Age Range", "age"), ("Gender", "gender"), ("Marital Status", "marital_status")
    ]

    for label, col in display_configs:
        if col in df.columns:
            df_plot = df[~df[col].astype(str).str.lower().isin(['u', 'unknown', 'nan', 'none', '', 'other'])]
            grp = df_plot.groupby(col).agg(Buyers=('Order ID', 'nunique'), Revenue=('Total', 'sum')).reset_index()
            
            if not grp.empty:
                st.markdown(f"## {label}")
                if col == "region":
                    with st.expander("📍 View Regional Identity Map"):
                        st.write("**Northeast:** CT, MA, ME, NH, NJ, NY, PA, RI, VT")
                        st.write("**Midwest:** IA, IL, IN, KS, MI, MN, MO, ND, NE, OH, SD, WI")
                        st.write("**South:** AL, AR, DC, DE, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV")
                        st.write("**West:** AK, AZ, CA, CO, HI, ID, MT, NM, NV, OR, UT, WA, WY")

                # --- 🚨 NEW Altair layered chart forprettier pie and labels 🚨 ---
                base = alt.Chart(grp).encode(
                    theta=alt.Theta(field="Revenue", type="quantitative"),
                    order=alt.Order(field="Revenue", sort="descending")
                )

                # Arcs layer (legend=None removes the legend)
                arcs = base.mark_arc(innerRadius=80, stroke="#fff").encode(
                    color=alt.Color(f"{col}:N", scale=alt.Scale(scheme='tableau20'), legend=None),
                    tooltip=[col, alt.Tooltip('Revenue', format='$,.0f')]
                )

                # Text labels layer (fontSize increased and outside pie)
                text = base.mark_text(radiusOffset=30, fontSize=16, font='Outfit').encode(
                    text=f"{col}:N",
                    order=alt.Order(field="Revenue", sort="descending")
                )

                chart = alt.layer(arcs, text).properties(height=500)
                st.altair_chart(chart, use_container_width=True)
                
                # --- 🚨 UPDATED TABLE DISPLAY: Fit content and Green Gradient 🚨 ---
                # Add math inside the loop to ensure correct column usage
                grp['Revenue'] = pd.to_numeric(grp['Revenue'], errors='coerce').fillna(0)
                grp['% Share'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col: label})
                format_dict = {'Buyers': '{:,.0f}', 'Revenue': '${:,.2f}', '% Share': '{:.1f}%', 'AOV': '${:,.2f}'}
                
                # Apply green gradient to Percent Share to show ranking
                styler = grp.style.format(format_dict).background_gradient(subset=['% Share'], cmap=custom_light_green)
                render_premium_table(styler)
