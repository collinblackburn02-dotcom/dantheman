import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import matplotlib.colors as mcolors
import streamlit.components.v1 as components
import requests
from bs4 import BeautifulSoup

# ================ 1. Page Config & Dynamic Branding =================
st.set_page_config(page_title="Heavenly Heat | Audience Engine", page_icon="🪵", layout="wide", initial_sidebar_state="expanded")

# --- App State Management ---
if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"
if "df_icp" not in st.session_state: st.session_state.df_icp = None
if "df_visitors" not in st.session_state: st.session_state.df_visitors = None
if "brand_color" not in st.session_state: st.session_state.brand_color = "#B3845C"
if "brand_logo" not in st.session_state: st.session_state.brand_logo = None

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            
            /* === SLIMMER SIDEBAR === */
            [data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #E2D7C8; min-width: 275px !important; max-width: 275px !important; }}
            
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 600 !important; }}
            p, span, label, .stRadio label, .stTabs [data-baseweb="tab"] {{ color: #2D2421 !important; }}
            
            /* Sleek Buttons */
            div[data-testid="stButton"] button {{ border-radius: 8px; font-weight: 500; transition: all 0.2s ease-in-out; padding: 0px 10px !important; }}
            div[data-testid="stButton"] button, div[data-testid="stButton"] button p {{ font-size: 0.85rem !important; white-space: nowrap !important; overflow: visible !important; }}
            
            /* Primary Button styled with Dynamic Brand Color */
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; box-shadow: 0 4px 6px rgba(0,0,0, 0.1); }}
            div[data-testid="stButton"] button[kind="primary"] p {{ font-weight: 800 !important; color: #FFFFFF !important; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; }}
            
            /* Metric Cards */
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px 24px; box-shadow: 0 4px 10px rgba(45, 36, 33, 0.04); }}
            [data-testid="stMetricLabel"] {{ color: {primary_color} !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.85rem; }}
            [data-testid="stMetricValue"] {{ color: #2D2421 !important; font-weight: 700; font-size: 2.2rem; }}
            
            /* Tabs styling */
            .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
            .stTabs [data-baseweb="tab"] {{ font-weight: 600; padding-bottom: 10px; }}
            .stTabs [aria-selected="true"] {{ border-bottom: 3px solid {primary_color} !important; color: {primary_color} !important; }}
            
            /* === LUXURY HTML TABLE STYLING === */
            .premium-table-container {{ width: 100%; overflow-x: auto; border-radius: 12px; border: 1px solid #E2D7C8; box-shadow: 0 4px 10px rgba(45, 36, 33, 0.04); background: #FFFFFF; margin-bottom: 1rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; font-family: 'Outfit', sans-serif !important; margin-bottom: 0 !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; font-weight: 700 !important; text-align: center !important; padding: 12px 14px !important; border-bottom: 2px solid #D5C6B3 !important; text-transform: uppercase !important; font-size: 0.70rem !important; white-space: nowrap !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 10px 14px !important; border-bottom: 1px solid #F0EAD6 !important; color: #3A2A26 !important; font-size: 0.80rem !important; white-space: nowrap !important; }}
            .premium-table-container td:first-child {{ font-weight: 700 !important; text-align: center !important; }}
            
            /* Hide sidebar toggle if onboarding */
            {'''[data-testid="collapsedControl"] { display: none; }''' if st.session_state.app_state == "onboarding" else ""}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(st.session_state.brand_color)

# Soft Green Colormap
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])
custom_tan_reversed = mcolors.LinearSegmentedColormap.from_list("custom_tan_r", ["#B3845C", "#E2D7C8", "#F9F7F3"])

def render_premium_table(styler_obj):
    try: styler_obj = styler_obj.hide(axis="index")
    except AttributeError: styler_obj = styler_obj.hide_index() 
    html = styler_obj.to_html()
    st.markdown(f'<div class="premium-table-container">{html}</div>', unsafe_allow_html=True)

# ================ 2. BigQuery Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict: creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_visitor_base():
    """Loads ONLY the total visitor counts from the old table to act as our baseline denominator."""
    client = get_bq_client()
    df = client.query("SELECT gender, age, income, region, net_worth, children, marital_status, homeowner, credit_rating, total_visitors FROM `xenon-mantis-430216-n4.final_dashboard.demographic_leaderboard`").to_dataframe()
    return df.fillna("Unknown").replace("", "Unknown")

configs = [("Gender", "gender"), ("Age", "age"), ("Income", "income"), ("Region", "region"), ("Net Worth", "net_worth"), ("Children", "children"), ("Marital Status", "marital_status"), ("Homeowner", "homeowner"), ("Credit Rating", "credit_rating")]

# ================ 3. STATE 1: ONBOARDING SCREEN =================
if st.session_state.app_state == "onboarding":
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-top: 50px;'>🎯 Audience Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 3rem;'>Upload your recent orders to reveal your Customer DNA.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 1. Brand Your App (Optional)")
        website_url = st.text_input("🔗 What is your website URL?", placeholder="https://www.yourstore.com")
        if website_url:
            if not website_url.startswith('http'): website_url = 'https://' + website_url
            with st.spinner("Extracting brand DNA..."):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    r = requests.get(website_url, headers=headers, timeout=5)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    color_meta = soup.find('meta', attrs={'name': 'theme-color'})
                    if color_meta:
                        st.session_state.brand_color = color_meta['content']
                        st.success(f"🎨 Brand color extracted: {st.session_state.brand_color}")
                        st.rerun()
                except Exception:
                    st.warning("Could not extract branding automatically. Using default.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 2. Upload Order Data")
        st.info("📦 **Required exactly:** `Date`, `Email`, `Order ID`, `Total`")
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            df_orders = pd.read_csv(uploaded_file)
            required_cols = ["Date", "Email", "Order ID", "Total"]
            missing_cols = [col for col in required_cols if col not in df_orders.columns]
            
            if missing_cols:
                st.error(f"⚠️ Your CSV is missing: **{', '.join(missing_cols)}**")
            else:
                with st.spinner("Pushing orders to BigQuery and matching pixels..."):
                    try:
                        client = get_bq_client()
                        project_id = dict(st.secrets["gcp_service_account"])["project_id"]
                        temp_table_id = f"{project_id}.final_dashboard.streamlit_temp_orders"
                        
                        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                        job = client.load_table_from_dataframe(df_orders, temp_table_id, job_config=job_config)
                        job.result()
                        
                        query = f"""
                            SELECT 
                                o.Email, o.`Order ID` as Order_ID, o.Total,
                                p.GENDER as gender, p.AGE_RANGE as age, p.INCOME_RANGE as income, 
                                CASE 
                                    WHEN p.PERSONAL_STATE IN ('CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA') THEN 'Northeast'
                                    WHEN p.PERSONAL_STATE IN ('IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI') THEN 'Midwest'
                                    WHEN p.PERSONAL_STATE IN ('AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'DC') THEN 'South'
                                    WHEN p.PERSONAL_STATE IN ('AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY') THEN 'West'
                                    ELSE 'Unknown'
                                END as region,
                                p.NET_WORTH as net_worth, p.CHILDREN as children, p.MARRIED as marital_status, 
                                p.HOMEOWNER as homeowner, p.SKIPTRACE_CREDIT_RATING as credit_rating
                            FROM `{temp_table_id}` o
                            LEFT JOIN `xenon-mantis-430216-n4.visitors_raw.all_visitors_combined` p
                            ON LOWER(p.PERSONAL_EMAILS) LIKE CONCAT('%', LOWER(o.Email), '%') 
                               OR LOWER(p.BUSINESS_EMAIL) = LOWER(o.Email)
                        """
                        df_joined = client.query(query).to_dataframe()
                        
                        # Clean currency and convert to numeric
                        df_joined['Total'] = df_joined['Total'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        df_joined['Total'] = pd.to_numeric(df_joined['Total'], errors='coerce').fillna(0)
                        
                        # Clean demographic columns
                        demo_cols = [c[1] for c in configs]
                        for col in demo_cols:
                            if col in df_joined.columns:
                                df_joined[col] = df_joined[col].astype(str).replace(['nan', 'None', '<NA>', ''], 'Unknown')
                        
                        # Save to memory and unlock dashboard
                        st.session_state.df_icp = df_joined
                        st.session_state.df_visitors = load_visitor_base()
                        st.session_state.app_state = "dashboard"
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

# ================ 4. STATE 2: LIVE DASHBOARD =================
elif st.session_state.app_state == "dashboard":
    
    # --- Sidebar ---
    with st.sidebar:
        if st.session_state.brand_logo:
            st.image(st.session_state.brand_logo, use_container_width=True)
        else:
            st.markdown(f"<h2 style='color: {st.session_state.brand_color}; text-align: center; margin-bottom: 0;'>🪵 Heavenly Heat</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem;'>Intelligence Engine</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Global Controls")
        sort_order = st.radio("Ranking Order", ["High to Low", "Low to High"], horizontal=True)
        is_ascending = (sort_order == "Low to High")
        metric_choice = st.radio("Primary Metric Leaderboard", ["Rev/Visitor", "Conv %", "Revenue", "Purchases", "Visitors"])
        min_visitors = st.number_input("Minimum Traffic Floor", value=250)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔄 Upload New Orders", use_container_width=True): 
            st.session_state.app_state = "onboarding"
            st.session_state.brand_color = "#B3845C" # Reset to default
            st.rerun()

    metric_map = {"Conv %": "Conv %", "Purchases": "Purchases", "Revenue": "Revenue", "Visitors": "Visitors", "Rev/Visitor": "Rev/Visitor"}

    tab1, tab2 = st.tabs(["📊 Conversion Insights", "🧬 Customer DNA (ICP)"])

    # ---------------- TAB 1: CONVERSION INSIGHTS (Dynamic) ----------------
    with tab1:
        st.markdown('<p style="font-size: 2rem; font-weight: 700; margin-bottom: 0px;">Audience Insights Engine</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666; margin-top: -5px; margin-bottom: 30px;">Traffic and Conversion Optimization</p>', unsafe_allow_html=True)
        
        st.subheader("🔍 Single Variable Deep Dive")
        if "active_single_var" not in st.session_state: st.session_state.active_single_var = "Gender"
        var_cols = st.columns(len(configs))
        for i, (label, col_name) in enumerate(configs):
            if var_cols[i].button(label, key=f"btn_{label}_t1", type="primary" if st.session_state.active_single_var == label else "secondary", use_container_width=True):
                st.session_state.active_single_var = label
                st.rerun()
                
        selected_col = dict(configs)[st.session_state.active_single_var]
        
        # Pull static visitors
        df_v = st.session_state.df_visitors[~st.session_state.df_visitors[selected_col].isin(['Unknown', 'U', ''])]
        df_v_grp = df_v.groupby(selected_col)['total_visitors'].sum().reset_index().rename(columns={'total_visitors': 'Visitors'})
        
        # Pull dynamic purchases from uploaded CSV
        df_p = st.session_state.df_icp[~st.session_state.df_icp[selected_col].isin(['Unknown', 'U', ''])]
        df_p_grp = df_p.groupby(selected_col).agg(Purchases=('Email', 'nunique'), Revenue=('Total', 'sum')).reset_index()
        
        # Merge them together
        df_merged = pd.merge(df_v_grp, df_p_grp, on=selected_col, how='left').fillna(0)

        if not df_merged.empty:
            df_merged['Conv %'] = (df_merged['Purchases'] / df_merged['Visitors'] * 100).round(2)
            df_merged['Rev/Visitor'] = (df_merged['Revenue'] / df_merged['Visitors']).round(2)
            df_merged = df_merged[df_merged['Visitors'] >= min_visitors].sort_values(metric_map[metric_choice], ascending=is_ascending)
            display_df = df_merged.rename(columns={selected_col: st.session_state.active_single_var})
            
            styler = display_df.style.format({'Visitors': '{:,.0f}', 'Purchases': '{:,.0f}', 'Revenue': '${:,.2f}', 'Conv %': '{:.2f}%', 'Rev/Visitor': '${:,.2f}'}).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap=custom_light_green)
            render_premium_table(styler)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Predictive Power Drivers (Dynamic)
        st.subheader("🏆 Top Conversion Drivers")
        predictive_data = []
        for label, col_name in configs:
            df_v_sub = st.session_state.df_visitors[~st.session_state.df_visitors[col_name].isin(['Unknown', 'U', ''])]
            grp_v = df_v_sub.groupby(col_name)['total_visitors'].sum().reset_index()
            
            df_p_sub = st.session_state.df_icp[~st.session_state.df_icp[col_name].isin(['Unknown', 'U', ''])]
            grp_p = df_p_sub.groupby(col_name).agg(Purchases=('Email', 'nunique')).reset_index()
            
            grp = pd.merge(grp_v, grp_p, on=col_name, how='left').fillna(0).rename(columns={'total_visitors': 'Visitors'})
            grp = grp[grp['Visitors'] >= min_visitors]
            
            if len(grp) >= 2:
                grp['Conv %'] = (grp['Purchases'] / grp['Visitors']) * 100
                top_row, bot_row = grp.loc[grp['Conv %'].idxmax()], grp.loc[grp['Conv %'].idxmin()]
                predictive_data.append({"Demographic Trait": label, "Top Segment": top_row[col_name], "Conv % (Top)": top_row['Conv %'], "Worst Segment": bot_row[col_name], "Conv % (Worst)": bot_row['Conv %'], "Predictive Swing": top_row['Conv %'] - bot_row['Conv %']})

        if predictive_data:
            pred_df = pd.DataFrame(predictive_data).sort_values("Predictive Swing", ascending=is_ascending)
            styler = pred_df.style.format({'Conv % (Top)': '{:.2f}%', 'Conv % (Worst)': '{:.2f}%', 'Predictive Swing': '{:.2f}%'}).background_gradient(subset=['Predictive Swing', 'Conv % (Top)'], cmap=custom_light_green).background_gradient(subset=['Conv % (Worst)'], cmap=custom_tan_reversed)
            render_premium_table(styler)

    # ---------------- TAB 2: CUSTOMER DNA (ICP) ----------------
    with tab2:
        st.markdown('<p style="font-size: 2rem; font-weight: 700; margin-bottom: 0px;">Ideal Customer Profile (ICP)</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666; margin-top: -5px; margin-bottom: 30px;">Demographic DNA of your actual paying customers.</p>', unsafe_allow_html=True)
        
        df_joined = st.session_state.df_icp
        
        # KPIs
        total_buyers = df_joined['Email'].nunique()
        total_rev = df_joined['Total'].sum()
        overall_aov = total_rev / total_buyers if total_buyers > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Matched Buyers", f"{total_buyers:,.0f}")
        m2.metric("Total Attributed Revenue", f"${total_rev:,.2f}")
        m3.metric("Overall Average Order Value", f"${overall_aov:,.2f}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Distribution Tables
        dash_col1, dash_col2 = st.columns(2)
        for index, (label, col_name) in enumerate(configs):
            grp = df_joined.groupby(col_name).agg(Buyers=('Email', 'nunique'), Revenue=('Total', 'sum')).reset_index()
            grp = grp[grp[col_name] != "Unknown"]
            
            if not grp.empty:
                grp['% of Buyers'] = (grp['Buyers'] / grp['Buyers'].sum()) * 100
                grp['% of Revenue'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                grp['AOV'] = grp['Revenue'] / grp['Buyers']
                grp = grp.sort_values('Revenue', ascending=False).rename(columns={col_name: label})
                
                format_dict = {'Buyers': '{:,.0f}', '% of Buyers': '{:.1f}%', 'Revenue': '${:,.2f}', '% of Revenue': '{:.1f}%', 'AOV': '${:,.2f}'}
                styler = grp.style.format(format_dict).background_gradient(subset=['% of Revenue', '% of Buyers'], cmap=custom_light_green)
                
                with dash_col1 if index % 2 == 0 else dash_col2:
                    st.subheader(f"{label} Distribution")
                    render_premium_table(styler)

# Prevent auto-scroll
components.html("<script>setTimeout(function() { window.parent.document.querySelector('.main').scrollTo(0, 0); }, 100);</script>", height=0)
