import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import itertools 

# ================ Setup =================
st.set_page_config(page_title="Heavenly Insights", page_icon="🪵", layout="wide")

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

# ================ Global Controls =================
with st.sidebar:
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric", ["Rev/Visitor", "Conv %", "Revenue", "Purchases", "Visitors"])
    min_visitors = st.number_input("Traffic Floor", value=250)
    if st.button("Reset Filters"): st.rerun()

st.title("🪵 Audience Insights Engine")
metric_map = {"Conv %": "Conv %", "Purchases": "total_purchasers", "Revenue": "total_revenue", "Visitors": "total_visitors", "Rev/Visitor": "Rev/Visitor"}

# ================ 1. Single Variable Deep Dive =================
st.subheader("🔍 Single Variable Deep Dive")
single_var_options = {"Gender": "gender", "Age": "age", "Income": "income", "Region": "region", "Net Worth": "net_worth", "Children": "children", "Marital Status": "marital_status", "Homeowner": "homeowner", "Credit Rating": "credit_rating"}

if "active_single_var" not in st.session_state: st.session_state.active_single_var = "Gender"
var_cols = st.columns(len(single_var_options))
for i, label in enumerate(single_var_options.keys()):
    if var_cols[i].button(label, key=f"btn_{label}", type="primary" if st.session_state.active_single_var == label else "secondary", use_container_width=True):
        st.session_state.active_single_var = label
        st.rerun()

selected_col = single_var_options[st.session_state.active_single_var]
df_single = df_master.groupby([selected_col]).agg(total_visitors=('total_visitors', 'sum'), total_purchasers=('total_purchasers', 'sum'), total_revenue=('total_revenue', 'sum')).reset_index()

if not df_single.empty:
    df_single['Conv %'] = (df_single['total_purchasers'] / df_single['total_visitors'] * 100).round(2)
    df_single['Rev/Visitor'] = (df_single['total_revenue'] / df_single['total_visitors']).round(2)
    df_single = df_single[df_single['total_visitors'] >= min_visitors]
    
    # --- LOGICAL SORTING ---
    if selected_col == 'income':
        sort_order = ['$0-$59,999', '$60,000-$99,999', '$100,000-$199,999', '$200,000+']
        df_single[selected_col] = pd.Categorical(df_single[selected_col], categories=sort_order, ordered=True)
    elif selected_col == 'net_worth':
        sort_order = ['$49,999 and below', '$50,000-$99,999', '$100,000-$249,999', '$250,000-$499,999', '$500,000-$999,999', '$1,000,000+']
        df_single[selected_col] = pd.Categorical(df_single[selected_col], categories=sort_order, ordered=True)
    elif selected_col == 'credit_rating':
        sort_order = ['High (A, B, C)', 'Medium (D, E)', 'Low (F, G)']
        df_single[selected_col] = pd.Categorical(df_single[selected_col], categories=sort_order, ordered=True)
    
    # Sort by either Category order or Metric
    is_bucketed = selected_col in ['income', 'net_worth', 'credit_rating']
    df_single = df_single.sort_values(selected_col if is_bucketed else metric_map[metric_choice], ascending=not is_bucketed)

    st.dataframe(df_single.style.format({'Conv %': '{:.2f}%', 'total_revenue': '${:,.2f}', 'Rev/Visitor': '${:,.2f}'}), use_container_width=True)

st.markdown("---")

# ================ 2. Multi-Variable Combination Analysis =================
st.subheader("📊 Multi-Variable Combination Analysis")
configs = [("Gender", "gender"), ("Age", "age"), ("Income", "income"), ("Region", "region"), ("Net Worth", "net_worth"), ("Children", "children"), ("Marital Status", "marital_status"), ("Homeowner", "homeowner"), ("Credit Rating", "credit_rating")]
selected_filters, included_types = {}, []
filter_cols = st.columns(3)

for i, (label, col_name) in enumerate(configs):
    with filter_cols[i % 3]:
        with st.container(border=True):
            is_inc = st.checkbox(f"Inc {label}", key=f"inc_{col_name}")
            opts = sorted(df_master[col_name].unique().tolist())
            # Ensure filters match the logical bucket order
            if col_name == 'income': opts = ['$0-$59,999', '$60,000-$99,999', '$100,000-$199,999', '$200,000+']
            elif col_name == 'net_worth': opts = ['$49,999 and below', '$50,000-$99,999', '$100,000-$249,999', '$250,000-$499,999', '$500,000-$999,999', '$1,000,000+']
            elif col_name == 'credit_rating': opts = ['High (A, B, C)', 'Medium (D, E)', 'Low (F, G)']

            val = st.multiselect(f"Filter {label}", opts, key=f"f_{col_name}")
            if is_inc: included_types.append(col_name)
            if val: selected_filters[col_name] = val

dff = df_master.copy()
for col, vals in selected_filters.items(): dff = dff[dff[col].isin(vals)]

if included_types and not dff.empty:
    combos = []
    for r in range(1, min(3, len(included_types)) + 1):
        for subset in itertools.combinations(included_types, r):
            grp = dff.groupby(list(subset)).agg(total_visitors=('total_visitors', 'sum'), total_purchasers=('total_purchasers', 'sum'), total_revenue=('total_revenue', 'sum')).reset_index()
            combos.append(grp)
    if combos:
        res = pd.concat(combos, ignore_index=True).drop_duplicates(subset=included_types).fillna("")
        res['Conv %'] = (res['total_purchasers'] / res['total_visitors'] * 100).round(2)
        res['Rev/Visitor'] = (res['total_revenue'] / res['total_visitors']).round(2)
        st.dataframe(res[res['total_visitors'] >= min_visitors].sort_values(metric_map[metric_choice], ascending=False), use_container_width=True)

# ================ 3. AI Data Agent =================
st.markdown("---")
st.subheader("🤖 Heavenly AI Data Agent")
if "GEMINI_API_KEY" in st.secrets:
    from pandasai import SmartDataframe
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
    sdf = SmartDataframe(df_master, config={"llm": llm})
    if prompt := st.chat_input("Ask me about your audience..."):
        with st.chat_message("assistant"): st.markdown(sdf.chat(prompt))
