import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import itertools 

# ================ Brand palette & CSS =================
BRAND = {
    "bg": "#FAF8F5",       
    "fg": "#3A2A26",       
    "accent": "#8C6239",   
    "card": "#FFFFFF"      
}

def inject_css():
    st.markdown(f"""
        <style>
            .stApp {{ background: {BRAND["bg"]}; color: {BRAND["fg"]}; }}
            .stDataFrame {{ border-radius: 12px; background: {BRAND["card"]}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            .attr-title {{ font-weight: 800; color: {BRAND["accent"]}; font-size: 0.95rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
            [data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid rgba(140, 98, 57, 0.1); }}
            hr {{ border-top: 1px solid rgba(140, 98, 57, 0.2); margin-top: 2rem; margin-bottom: 2rem; }}
            
            div[data-testid="stButton"] button[kind="primary"] {{
                background-color: {BRAND["accent"]};
                color: white;
                font-weight: 800;
                border: 1px solid {BRAND["accent"]};
                transition: all 0.2s ease-in-out;
            }}
            div[data-testid="stButton"] button[kind="secondary"] {{
                background-color: {BRAND["card"]};
                color: {BRAND["fg"]};
                border: 1px solid rgba(140, 98, 57, 0.2);
                transition: all 0.2s ease-in-out;
            }}
            div[data-testid="stButton"] button[kind="secondary"]:hover {{
                border: 1px solid {BRAND["accent"]};
                color: {BRAND["accent"]};
            }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Insights", page_icon="🪵", layout="wide")
inject_css()

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
else:
    st.sidebar.markdown(f"<h2 style='color: {BRAND['accent']};'>🪵 Heavenly Heat</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# ================ Connection =================
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
    return df.fillna("")

df_master = load_data()

# ================ Sidebar & Global Controls =================
with st.sidebar:
    st.header("Global Controls")
    # NEW: Rev/Visitor takes the top spot!
    metric_choice = st.radio("Primary Metric", ["Rev/Visitor", "Conv %", "Revenue", "Purchases", "Visitors"])
    min_visitors = st.number_input("Traffic Floor", value=250)
    st.markdown("---")
    if st.button("Reset Filters"):
        st.rerun()

st.title("🪵 Audience Insights Engine")

# ================ 1. UI: Checkboxes and Dropdowns =================
cols = st.columns(3)
configs = [
    ("Gender", "gender"), ("Age", "age"), ("Income", "income"),
    ("State", "state"), ("Net Worth", "net_worth"), ("Children", "children"),
    ("Credit Rating", "credit_rating")
]

selected_filters = {}
included_types = []

for i, (label, col_name) in enumerate(configs):
    with cols[i % 3]:
        with st.container(border=True):
            c_title, c_inc = st.columns([3, 1])
            c_title.markdown(f'<p class="attr-title">{label}</p>', unsafe_allow_html=True)
            
            is_inc = c_inc.checkbox("Inc", value=(i<3), key=f"inc_{col_name}")
            valid_opts = sorted([x for x in df_master[col_name].unique() if x != ""])
            val = st.selectbox(f"Filter {label}", ["- All -"] + valid_opts, key=f"filter_{col_name}", label_visibility="collapsed")
            
            if is_inc: included_types.append(col_name)
            if val != "- All -": selected_filters[col_name] = val

# ================ 2. NEW DYNAMIC LOGIC: The Data Cube =================
dff = df_master.copy()

for col, val in selected_filters.items():
    if col in dff.columns:
        dff = dff[dff[col] == val]

metric_map = {
    "Conv %": "Conv %", 
    "Purchases": "total_purchasers", 
    "Revenue": "total_revenue", 
    "Visitors": "total_visitors",
    "Rev/Visitor": "Rev/Visitor" # Added to mapping
}

if included_types and not dff.empty:
    all_combos_dfs = []
    max_combo_size = min(3, len(included_types))
    
    if len(included_types) > 3:
        st.info("💡 **Compute Saver Active:** You selected more than 3 variables. The table is automatically capping combination groups at a maximum of 3 to keep the dashboard lightning fast!")
    
    for r in range(1, max_combo_size + 1):
        for subset in itertools.combinations(included_types, r):
            subset = list(subset)
            
            temp_df = dff.copy()
            
            for col in subset:
                temp_df = temp_df[temp_df[col] != ""]
                
            if temp_df.empty:
                continue
                
            grp = temp_df.groupby(subset).agg(
                total_visitors=('total_visitors', 'sum'),
                total_purchasers=('total_purchasers', 'sum'),
                total_revenue=('total_revenue', 'sum') 
            ).reset_index()
            
            for col in included_types:
                if col not in subset:
                    grp[col] = ""
                    
            all_combos_dfs.append(grp)
            
    if all_combos_dfs:
        dff = pd.concat(all_combos_dfs, ignore_index=True)
        
        # Calculate the metrics!
        dff['Conv %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
        dff['Rev/Visitor'] = (dff['total_revenue'] / dff['total_visitors']).round(2)
        
        dff = dff[dff['total_visitors'] >= min_visitors]
        
        if dff.empty:
            st.warning("No combinations met the Traffic Floor minimum.")
        else:
            dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
            dff.index += 1
            
            final_display_cols = included_types + ["total_visitors", "total_purchasers", "Conv %", "total_revenue", "Rev/Visitor"]
            
            st.dataframe(
                dff[final_display_cols].rename(columns={
                    "gender": "Gender", "age": "Age", "income": "Income", 
                    "state": "State", "net_worth": "Net Worth", "children": "Children", "credit_rating": "Credit Rating",
                    "total_visitors": "Visitors", "total_purchasers": "Purchases", "total_revenue": "Revenue"
                }).style.format({
                    'Conv %': '{:.2f}%', 
                    'Revenue': '${:,.2f}',
                    'Rev/Visitor': '${:,.2f}'
                }).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap='YlGn'),
                use_container_width=True
            )
    else:
        st.warning("No data found for these combinations.")
elif not included_types:
    st.info("Please check at least one 'Inc' box to group your audience data.")

# ================ 3. NEW DYNAMIC LOGIC: Single Variable Deep Dive =================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🔍 Single Variable Deep Dive")

single_var_options = {
    "Gender": "gender",
    "Age": "age",
    "Income": "income",
    "State": "state",
    "Net Worth": "net_worth",
    "Children": "children",
    "Credit Rating": "credit_rating"
}

if "active_single_var" not in st.session_state:
    st.session_state.active_single_var = "Gender"

var_cols = st.columns(len(single_var_options))

for i, label in enumerate(single_var_options.keys()):
    btn_type = "primary" if st.session_state.active_single_var == label else "secondary"
    if var_cols[i].button(label, key=f"btn_{label}", type=btn_type, use_container_width=True):
        st.session_state.active_single_var = label
        st.rerun()

selected_single_label = st.session_state.active_single_var
selected_single_col = single_var_options[selected_single_label]

df_clean_single = df_master[df_master[selected_single_col] != ""]

df_single = df_clean_single.groupby([selected_single_col]).agg(
    total_visitors=('total_visitors', 'sum'),
    total_purchasers=('total_purchasers', 'sum'),
    total_revenue=('total_revenue', 'sum')
).reset_index()

if not df_single.empty:
    df_single['Conv %'] = (df_single['total_purchasers'] / df_single['total_visitors'] * 100).round(2)
    df_single['Rev/Visitor'] = (df_single['total_revenue'] / df_single['total_visitors']).round(2)
    
    df_single = df_single[df_single['total_visitors'] >= min_visitors]
    
    if df_single.empty:
        st.info(f"No groups within **{selected_single_label}** met the Traffic Floor minimum of {min_visitors}.")
    else:
        df_single = df_single.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
        df_single.index += 1
        
        display_cols = [selected_single_col, "total_visitors", "total_purchasers", "Conv %", "total_revenue", "Rev/Visitor"]
        
        st.dataframe(
            df_single[display_cols].rename(columns={
                selected_single_col: selected_single_label,
                "total_visitors": "Visitors", 
                "total_purchasers": "Purchases",
                "total_revenue": "Revenue"
            }).style.format({
                'Conv %': '{:.2f}%', 
                'Revenue': '${:,.2f}',
                'Rev/Visitor': '${:,.2f}'
            }).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap='YlGn'),
            use_container_width=True
        )
else:
    st.info("No data available for this variable.")

# ================ 4. AI Data Agent (Gemini) =================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🤖 Heavenly AI Data Agent")

if "GEMINI_API_KEY" not in st.secrets:
    st.warning("⚠️ **API Key Missing:** Please add your `GEMINI_API_KEY` to your Streamlit secrets to wake up the AI agent.")
else:
    from pandasai import SmartDataframe
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    sdf = SmartDataframe(df_master, config={"llm": llm})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about your audience data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Crunching the numbers..."):
                try:
                    response = sdf.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"I hit a snag trying to calculate that: {e}")
