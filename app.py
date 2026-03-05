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
            span[data-baseweb="tag"] {
                background-color: #C1A68D !important;
                color: #FFFFFF !important;
            }
            
            [data-testid="stExpander"], .st-emotion-cache-1z1q1o0 { border: 1px solid #E2D7C8 !important; border-radius: 12px !important; background: #FFFFFF; box-shadow: 0 2px 4px rgba(45, 36, 33, 0.02); }
            .stDataFrame { border: 1px solid #E2D7C8; border-radius: 12px; overflow: hidden; }
            hr { border-top: 1px solid rgba(158, 96, 54, 0.2); margin-top: 2rem; margin-bottom: 2rem; }
            .brand-header { font-size: 2.5rem; font-weight: 700; color: #2D2421; margin-bottom: 0px; padding-bottom: 0px; }
            .brand-subtitle { color: #B3845C; font-weight: 500; font-size: 1.1rem; margin-top: -5px; margin-bottom: 30px; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# Custom Light Green Colormap
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

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

# Header Area
st.markdown('<p class="brand-header">Audience Insights Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Powered by Heavenly Heat Data Infrastructure</p>', unsafe_allow_html=True)

# ================ 4. Single Variable Deep Dive =================
st.subheader("🔍 Single Variable Deep Dive")
single_var_options = {"Gender": "gender", "Age": "age", "Income": "income", "Region": "region", "Net Worth": "net_worth", "Children": "children", "Marital Status": "marital_status", "Homeowner": "homeowner", "Credit Rating": "credit_rating"}

if "active_single_var" not in st.session_state: st.session_state.active_single_var = "Gender"
var_cols = st.columns(len(single_var_options))
for i, label in enumerate(single_var_options.keys()):
    if var_cols[i].button(label, key=f"btn_{label}", type="primary" if st.session_state.active_single_var == label else "secondary", use_container_width=True):
        st.session_state.active_single_var = label
        st.rerun()

selected_col = single_var_options[st.session_state.active_single_var]

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
    
    # THE FIX: Always sort strictly by the user's chosen Primary Metric, highest to lowest.
    df_single = df_single.sort_values(metric_map[metric_choice], ascending=False)
    
    display_df = df_single.rename(columns={selected_col: st.session_state.active_single_var})
    
    st.dataframe(
        display_df.style.format({'Conv %': '{:.2f}%', 'Revenue': '${:,.2f}', 'Rev/Visitor': '${:,.2f}'}).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap=custom_light_green), 
        use_container_width=True, 
        hide_index=True
    )
else:
    st.info("No data available for this variable with the current traffic floor.")

st.markdown("<hr>", unsafe_allow_html=True)

# ================ 5. Multi-Variable Combination Analysis =================
st.subheader("📊 Multi-Variable Combination Matrix")

with st.expander("🎛️ Combination Filters", expanded=True):
    st.markdown("<p style='font-size: 0.9rem; color: #666;'>Filters applied here ONLY affect the Combination Matrix below.</p>", unsafe_allow_html=True)
    
    configs = [("Gender", "gender"), ("Age", "age"), ("Income", "income"), ("Region", "region"), ("Net Worth", "net_worth"), ("Children", "children"), ("Marital Status", "marital_status"), ("Homeowner", "homeowner"), ("Credit Rating", "credit_rating")]
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

dff = df_master.copy()
for col, vals in selected_filters.items(): 
    dff = dff[dff[col].isin(vals)]

if included_types and not dff.empty:
    combos = []
    max_combo_size = min(3, len(included_types))
    
    for r in range(1, max_combo_size + 1):
        for subset in itertools.combinations(included_types, r):
            temp_df = dff.copy()
            
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
            st.dataframe(
                final_res[ordered_cols].rename(columns=rename_dict).style.format({'Conv %': '{:.2f}%', 'Revenue': '${:,.2f}', 'Rev/Visitor': '${:,.2f}'}).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap=custom_light_green), 
                use_container_width=True, 
                hide_index=True
            )
elif not included_types:
    st.info("👆 Check the 'Inc' boxes above to build your combination matrix, or view your baseline totals below:")
    
    # Calculate the summary stats based on the current dropdown filters
    total_vis = dff['total_visitors'].sum()
    total_purch = dff['total_purchasers'].sum()
    total_rev = dff['total_revenue'].sum()
    
    if total_vis >= min_visitors:
        summary_df = pd.DataFrame([{
            "Audience Segment": "Overall Filtered Baseline",
            "Visitors": total_vis, 
            "Purchases": total_purch, 
            "Revenue": total_rev,
            "Conv %": (total_purch / total_vis * 100).round(2) if total_vis > 0 else 0,
            "Rev/Visitor": (total_rev / total_vis).round(2) if total_vis > 0 else 0
        }])
        
        st.dataframe(
            summary_df.style.format({'Conv %': '{:.2f}%', 'Revenue': '${:,.2f}', 'Rev/Visitor': '${:,.2f}'}), 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(f"Not enough traffic to meet the Minimum Floor of {min_visitors}.")

# ================ 6. AI Data Agent =================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🤖 Heavenly AI Data Agent")
if "GEMINI_API_KEY" in st.secrets:
    from pandasai import SmartDataframe
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
    sdf = SmartDataframe(df_master, config={"llm": llm})
    if prompt := st.chat_input("Ask me about your audience..."):
        with st.chat_message("assistant"): st.markdown(sdf.chat(prompt))
