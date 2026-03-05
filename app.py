import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import itertools
import matplotlib.colors as mcolors

# ================ 1. Page Config & Premium Brand CSS =================
st.set_page_config(page_title="Heavenly Heat | Insights", page_icon="🪵", layout="wide", initial_sidebar_state="expanded")

def apply_custom_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
            .stApp { background-color: #F9F7F3; }
            [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2D7C8; }
            h1, h2, h3 { color: #2D2421 !important; font-weight: 600 !important; }
            p, span, label, .stRadio label { color: #2D2421 !important; }
            
            /* Sleek Buttons */
            div[data-testid="stButton"] button { border-radius: 8px; font-weight: 500; transition: all 0.2s ease-in-out; }
            
            /* The SELECTED Variable Button - Lighter Tan & Bold */
            div[data-testid="stButton"] button[kind="primary"] { 
                background-color: #B3845C !important; 
                color: #FFFFFF !important; 
                font-weight: 800 !important; 
                border: none; 
                box-shadow: 0 4px 6px rgba(179, 132, 92, 0.2); 
            }
            div[data-testid="stButton"] button[kind="primary"]:hover { background-color: #9C6F49 !important; }
            
            /* Unselected Buttons */
            div[data-testid="stButton"] button[kind="secondary"] { background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; }
            div[data-testid="stButton"] button[kind="secondary"]:hover { border-color: #B3845C; color: #B3845C; }
            
            /* Overriding Streamlit's Default Red in Dropdown Tags to Dark Tan */
            span[data-baseweb="tag"] { background-color: #C1A68D !important; color: #FFFFFF !important; }
            
            /* Metric Cards */
            [data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #E2D7C8;
                border-radius: 12px;
                padding: 20px 24px;
                box-shadow: 0 4px 10px rgba(45, 36, 33, 0.04);
            }
            [data-testid="stMetricLabel"] { color: #B3845C !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.85rem; }
            [data-testid="stMetricValue"] { color: #2D2421 !important; font-weight: 700; font-size: 2.2rem; }
            
            /* Expanders & Dividers */
            [data-testid="stExpander"], .st-emotion-cache-1z1q1o0 { border: 1px solid #E2D7C8 !important; border-radius: 12px !important; background: #FFFFFF; box-shadow: 0 2px 4px rgba(45, 36, 33, 0.02); }
            hr { border-top: 1px solid rgba(158, 96, 54, 0.2); margin-top: 2rem; margin-bottom: 2rem; }
            
            /* Headers */
            .brand-header { font-size: 2.5rem; font-weight: 700; color: #2D2421; margin-bottom: 0px; padding-bottom: 0px; }
            .brand-subtitle { color: #B3845C; font-weight: 500; font-size: 1.1rem; margin-top: -5px; margin-bottom: 30px; }
            
            /* === NEW: LUXURY HTML TABLE STYLING === */
            .premium-table-container {
                width: 100%;
                overflow-x: auto;
                border-radius: 12px;
                border: 1px solid #E2D7C8;
                box-shadow: 0 4px 10px rgba(45, 36, 33, 0.04);
                background: #FFFFFF;
                margin-bottom: 1rem;
            }
            .premium-table-container table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Outfit', sans-serif;
            }
            .premium-table-container th {
                background-color: #F2EBE1 !important;
                color: #9E6036 !important;
                font-weight: 700 !important;
                text-align: center !important;
                padding: 14px 16px !important;
                border-bottom: 2px solid #D5C6B3 !important;
                text-transform: uppercase;
                font-size: 0.85rem;
                letter-spacing: 0.5px;
            }
            .premium-table-container td {
                text-align: center !important;
                padding: 12px 16px !important;
                border-bottom: 1px solid #F0EAD6 !important;
                color: #3A2A26 !important;
                font-size: 0.95rem;
                vertical-align: middle;
            }
            .premium-table-container tr:last-child td { border-bottom: none !important; }
            .premium-table-container tr:hover { opacity: 0.95; }
            
            /* Bold & Left-Align the first column to anchor the data */
            .premium-table-container td:first-child {
                font-weight: 700 !important;
                color: #2D2421 !important;
                text-align: left !important;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# Premium Custom Colormaps 
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])
custom_tan_reversed = mcolors.LinearSegmentedColormap.from_list("custom_tan_r", ["#B3845C", "#E2D7C8", "#F9F7F3"])

# Helper function to render perfect HTML tables
def render_premium_table(styler_obj):
    try:
        styler_obj = styler_obj.hide(axis="index")
    except AttributeError:
        styler_obj = styler_obj.hide_index() # Fallback for older pandas versions
    html = styler_obj.to_html()
    st.markdown(f'<div class="premium-table-container">{html}</div>', unsafe_allow_html=True)

# Reference String for Regions
REGION_INFO = "📍 **Region Breakdown:** **Northeast:** CT, ME, MA, NH, RI, VT, NJ, NY, PA | **Midwest:** IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI | **South:** AL, AR, DE, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV, DC | **West:** AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY"

# Formatting dictionaries (Now with commas for whole numbers!)
format_standard = {'Visitors': '{:,.0f}', 'Purchases': '{:,.0f}', 'Revenue': '${:,.2f}', 'Conv %': '{:.2f}%', 'Rev/Visitor': '${:,.2f}'}
format_drivers = {'Conv % (Top)': '{:.2f}%', 'Conv % (Worst)': '{:.2f}%', 'Predictive Swing': '{:.2f}%'}

# ================ 2. Data Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_data():
    client = get_bq_client()
    df = client.query("SELECT * FROM `xenon-mantis-430216-n4.final_dashboard.demographic_leaderboard`").to_dataframe()
    return df.fillna("Unknown").replace("", "Unknown")

df_master = load_data()

INCOME_MAP = {'$0-$59,999': 1, '$60,000-$99,999': 2, '$100,000-$199,999': 3, '$200,000+': 4}
NET_WORTH_MAP = {'$49,999 and below': 1, '$50,000-$99,999': 2, '$100,000-$249,999': 3, '$250,000-$499,999': 4, '$500,000-$999,999': 5, '$1,000,000+': 6}
CREDIT_MAP = {'High (A, B, C)': 1, 'Medium (D, E)': 2, 'Low (F, G)': 3}

# ================ 3. Global Controls (Left Sidebar) =================
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("<h2 style='color: #B3845C; text-align: center; margin-bottom: 0;'>🪵 Heavenly Heat</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem;'>Intelligence Engine</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric Leaderboard", ["Rev/Visitor", "Conv %", "Revenue", "Purchases", "Visitors"])
    min_visitors = st.number_input("Minimum Traffic Floor", value=250)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset All Filters", use_container_width=True): 
        st.rerun()

metric_map = {"Conv %": "Conv %", "Purchases": "Purchases", "Revenue": "Revenue", "Visitors": "Visitors", "Rev/Visitor": "Rev/Visitor"}

st.markdown('<p class="brand-header">Audience Insights Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Powered by Heavenly Heat Data Infrastructure</p>', unsafe_allow_html=True)

configs = [("Gender", "gender"), ("Age", "age"), ("Income", "income"), ("Region", "region"), ("Net Worth", "net_worth"), ("Children", "children"), ("Marital Status", "marital_status"), ("Homeowner", "homeowner"), ("Credit Rating", "credit_rating")]

# ================ 4. Single Variable Deep Dive (TOP) =================
st.subheader("🔍 Single Variable Deep Dive")

if "active_single_var" not in st.session_state: st.session_state.active_single_var = "Gender"
var_cols = st.columns(len(configs))
for i, (label, col_name) in enumerate(configs):
    if var_cols[i].button(label, key=f"btn_{label}", type="primary" if st.session_state.active_single_var == label else "secondary", use_container_width=True):
        st.session_state.active_single_var = label
        st.rerun()

if st.session_state.active_single_var == "Region":
    st.info(REGION_INFO)

selected_col = dict(configs)[st.session_state.active_single_var]

df_clean_single = df_master[~df_master[selected_col].isin(['Unknown', 'U', ''])]

df_single = df_clean_single.groupby([selected_col]).agg(
    Visitors=('total_visitors', 'sum'), 
    Purchases=('total_purchasers', 'sum'), 
    Revenue=('total_revenue', 'sum')
).reset_index()

if not df_single.empty:
    df_single['Conv %'] = (df_single['Purchases'] / df_single['Visitors'] * 100).round(2)
    df_single['Rev/Visitor'] = (df_single['Revenue'] / df_single['Visitors']).round(2)
    df_single = df_single[df_single['Visitors'] >= min_visitors]
    
    df_single = df_single.sort_values(metric_map[metric_choice], ascending=False)
    display_df = df_single.rename(columns={selected_col: st.session_state.active_single_var})
    
    # Render with new Premium HTML logic
    styler = display_df.style.format(format_standard).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap=custom_light_green)
    render_premium_table(styler)
else:
    st.info("No data available for this variable with the current traffic floor.")

st.markdown("<hr>", unsafe_allow_html=True)

# ================ 5. Predictive Power Rankings (MIDDLE) =================
st.subheader("🏆 Top Conversion Drivers")
st.markdown("<p style='font-size: 0.95rem; color: #666;'>This table ranks demographic traits by their <b>Predictive Swing</b> (the conversion rate difference between a trait's best and worst performing segments).</p>", unsafe_allow_html=True)

predictive_data = []

for label, col_name in configs:
    df_clean = df_master[~df_master[col_name].isin(['Unknown', 'U', ''])]
    grp = df_clean.groupby(col_name).agg(
        Visitors=('total_visitors', 'sum'),
        Purchases=('total_purchasers', 'sum')
    ).reset_index()
    
    grp = grp[grp['Visitors'] >= min_visitors]
    
    if len(grp) >= 2:
        grp['Conv %'] = (grp['Purchases'] / grp['Visitors']) * 100
        
        top_row = grp.loc[grp['Conv %'].idxmax()]
        bot_row = grp.loc[grp['Conv %'].idxmin()]
        swing = top_row['Conv %'] - bot_row['Conv %']
        
        predictive_data.append({
            "Demographic Trait": label,
            "Top Segment": top_row[col_name],
            "Conv % (Top)": top_row['Conv %'],
            "Worst Segment": bot_row[col_name],
            "Conv % (Worst)": bot_row['Conv %'],
            "Predictive Swing": swing
        })

if predictive_data:
    pred_df = pd.DataFrame(predictive_data).sort_values("Predictive Swing", ascending=False)
    
    styler = pred_df.style.format(format_drivers)\
        .background_gradient(subset=['Predictive Swing', 'Conv % (Top)'], cmap=custom_light_green) \
        .background_gradient(subset=['Conv % (Worst)'], cmap=custom_tan_reversed)
        
    render_premium_table(styler)
else:
    st.info("Not enough data to calculate predictive swings based on your current Traffic Floor.")

st.markdown("<hr>", unsafe_allow_html=True)

# ================ 6. Multi-Variable Combination Analysis (BOTTOM) =================
st.subheader("📊 Multi-Variable Combination Matrix")

with st.expander("🎛️ Combination Filters", expanded=True):
    st.markdown("<p style='font-size: 0.9rem; color: #666;'>Select filters below. The KPI cards and Combination Matrix will instantly update based on your choices.</p>", unsafe_allow_html=True)
    
    selected_filters, included_types = {}, []
    filter_cols = st.columns(3)

    for i, (label, col_name) in enumerate(configs):
        with filter_cols[i % 3]:
            c_title, c_inc = st.columns([3, 1])
            c_title.markdown(f'<p style="font-weight: 600; color: #B3845C; margin-bottom: 0;">{label}</p>', unsafe_allow_html=True)
            is_inc = c_inc.checkbox("Inc", key=f"inc_{col_name}", help=f"Include {label} in Combination Matrix")
            
            opts = [x for x in df_master[col_name].unique() if x not in ['Unknown', 'U', '']]
            
            if col_name == 'income': opts = sorted(opts, key=lambda x: INCOME_MAP.get(x, 99))
            elif col_name == 'net_worth': opts = sorted(opts, key=lambda x: NET_WORTH_MAP.get(x, 99))
            elif col_name == 'credit_rating': opts = sorted(opts, key=lambda x: CREDIT_MAP.get(x, 99))
            else: opts = sorted(opts)

            val = st.multiselect(f"Filter {label}", opts, key=f"f_{col_name}", label_visibility="collapsed", placeholder="All")
            
            if is_inc: included_types.append(col_name)
            if val: selected_filters[col_name] = val

if "region" in included_types:
    st.info(REGION_INFO)

dff_matrix = df_master.copy()
for col, vals in selected_filters.items(): 
    dff_matrix = dff_matrix[dff_matrix[col].isin(vals)]

# === TOP KPI DASHBOARD ===
st.markdown("<br>", unsafe_allow_html=True)
if not dff_matrix.empty and (selected_filters or included_types):
    total_vis = dff_matrix['total_visitors'].sum()
    total_purch = dff_matrix['total_purchasers'].sum()
    total_rev = dff_matrix['total_revenue'].sum()
    avg_conv = (total_purch / total_vis * 100) if total_vis > 0 else 0
    avg_rev_vis = (total_rev / total_vis) if total_vis > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Segment Visitors", f"{total_vis:,.0f}")
    m2.metric("Segment Purchases", f"{total_purch:,.0f}")
    m3.metric("Segment Conv Rate", f"{avg_conv:.2f}%")
    m4.metric("Segment Rev / Visitor", f"${avg_rev_vis:,.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

# === COMBINATION TABLE ===
if included_types and not dff_matrix.empty:
    combos = []
    max_combo_size = min(3, len(included_types))
    
    for r in range(1, max_combo_size + 1):
        for subset in itertools.combinations(included_types, r):
            temp_df = dff_matrix.copy()
            
            for col in subset:
                temp_df = temp_df[~temp_df[col].isin(['Unknown', 'U', ''])]
                
            if temp_df.empty:
                continue
                
            grp = temp_df.groupby(list(subset)).agg(
                Visitors=('total_visitors', 'sum'), 
                Purchases=('total_purchasers', 'sum'), 
                Revenue=('total_revenue', 'sum')
            ).reset_index()
            
            for col in included_types:
                if col not in subset:
                    if col in selected_filters and selected_filters[col]:
                        grp[col] = ", ".join(selected_filters[col])
                    else:
                        grp[col] = ""
                    
            combos.append(grp)
            
    if combos:
        res = pd.concat(combos, ignore_index=True)
        res = res.drop_duplicates(subset=included_types)
        
        res['Conv %'] = (res['Purchases'] / res['Visitors'] * 100).round(2)
        res['Rev/Visitor'] = (res['Revenue'] / res['Visitors']).round(2)
        
        final_res = res[res['Visitors'] >= min_visitors].sort_values(metric_map[metric_choice], ascending=False)
        
        metrics = ["Visitors", "Purchases", "Revenue", "Conv %", "Rev/Visitor"]
        ordered_cols = included_types + metrics
        
        rename_dict = {"gender": "Gender", "age": "Age", "income": "Income", "region": "Region", "net_worth": "Net Worth", "children": "Children", "marital_status": "Marital Status", "homeowner": "Homeowner", "credit_rating": "Credit Rating"}
        
        if final_res.empty:
            st.warning(f"No combinations met the Traffic Floor minimum of {min_visitors}.")
        else:
            styler = final_res[ordered_cols].rename(columns=rename_dict).style.format(format_standard).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap=custom_light_green)
            render_premium_table(styler)
            
elif not included_types:
    st.info("👆 Check the 'Inc' boxes to build your combination matrix.")

# ================ 7. AI Data Agent =================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🤖 Heavenly AI Data Agent")
if "GEMINI_API_KEY" in st.secrets:
    from pandasai import SmartDataframe
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
    sdf = SmartDataframe(df_master, config={"llm": llm, "enforce_privacy": True, "enable_cache": True})
    
    st.info("💡 **Tip:** Ask specific questions like 'What is the total revenue for Females?'")
    if prompt := st.chat_input("Ask me about your audience..."):
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"): 
            with st.spinner("Crunching data..."):
                try:
                    response = sdf.chat(prompt)
                    if response is None or str(response).strip() == "":
                        st.warning("I couldn't calculate that. Try rephrasing your question.")
                    else:
                        st.markdown(response)
                except Exception as e:
                    st.error(f"Error: {e}")
