import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import itertools

# ================ Setup & Brand =================
BRAND = {"bg": "#FAF8F5", "fg": "#3A2A26", "accent": "#8C6239", "card": "#FFFFFF"}
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
    # Map blanks to Unknown for safe filtering
    return df.fillna("Unknown").replace("", "Unknown")

df_master = load_data()

# ================ Global Controls =================
with st.sidebar:
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric", ["Rev/Visitor", "Conv %", "Revenue", "Purchases", "Visitors"])
    min_visitors = st.number_input("Traffic Floor", value=250)
    if st.button("Reset Filters"): st.rerun()

st.title("🪵 Audience Insights Engine")
metric_map = {"Conv %": "Conv %", "Purchases": "Purchases", "Revenue": "Revenue", "Visitors": "Visitors", "Rev/Visitor": "Rev/Visitor"}

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

# Hide 'Unknown' only for the active variable
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
    
    # Bucket Sorting
    is_bucketed = selected_col in ['income', 'net_worth', 'credit_rating']
    if selected_col == 'income':
        df_single[selected_col] = pd.Categorical(df_single[selected_col], categories=['$0-$59,999', '$60,000-$99,999', '$100,000-$199,999', '$200,000+'], ordered=True)
    elif selected_col == 'net_worth':
        df_single[selected_col] = pd.Categorical(df_single[selected_col], categories=['$49,999 and below', '$50,000-$99,999', '$100,000-$249,999', '$250,000-$499,999', '$500,000-$999,999', '$1,000,000+'], ordered=True)
    elif selected_col == 'credit_rating':
        df_single[selected_col] = pd.Categorical(df_single[selected_col], categories=['High (A, B, C)', 'Medium (D, E)', 'Low (F, G)'], ordered=True)
    
    df_single = df_single.sort_values(selected_col if is_bucketed else metric_map[metric_choice], ascending=not is_bucketed)
    
    display_df = df_single.rename(columns={selected_col: st.session_state.active_single_var})
    st.dataframe(display_df.style.format({'Conv %': '{:.2f}%', 'Revenue': '${:,.2f}', 'Rev/Visitor': '${:,.2f}'}).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap='YlGn'), use_container_width=True)

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
            
            # Clean Dropdowns
            opts = [x for x in df_master[col_name].unique() if x not in ['Unknown', 'U', '']]
            if col_name == 'income': sort_order = ['$0-$59,999', '$60,000-$99,999', '$100,000-$199,999', '$200,000+']
            elif col_name == 'net_worth': sort_order = ['$49,999 and below', '$50,000-$99,999', '$100,000-$249,999', '$250,000-$499,999', '$500,000-$999,999', '$1,000,000+']
            elif col_name == 'credit_rating': sort_order = ['High (A, B, C)', 'Medium (D, E)', 'Low (F, G)']
            else: sort_order = sorted(opts)
            
            if col_name in ['income', 'net_worth', 'credit_rating']:
                opts = sorted(opts, key=lambda x: sort_order.index(x) if x in sort_order else 99)
            else:
                opts = sorted(opts)

            val = st.multiselect(f"Filter {label}", opts, key=f"f_{col_name}")
            if is_inc: included_types.append(col_name)
            if val: selected_filters[col_name] = val

dff = df_master.copy()
# Apply strict dropdown filters
for col, vals in selected_filters.items(): 
    dff = dff[dff[col].isin(vals)]

if included_types and not dff.empty:
    combos = []
    # Limit combinations to 3 levels deep to prevent the app from freezing
    max_combo_size = min(3, len(included_types))
    
    # THE RESTORED LOGIC: Loop through 1-variable, 2-variable, and 3-variable combinations
    for r in range(1, max_combo_size + 1):
        for subset in itertools.combinations(included_types, r):
            temp_df = dff.copy()
            
            # Remove "Unknown" ONLY for the columns currently being grouped
            for col in subset:
                temp_df = temp_df[~temp_df[col].isin(['Unknown', 'U', ''])]
                
            if temp_df.empty:
                continue
                
            grp = temp_df.groupby(list(subset)).agg(
                Visitors=('total_visitors', 'sum'), 
                Purchases=('total_purchasers', 'sum'), 
                Revenue=('total_revenue', 'sum')
            ).reset_index()
            
            # Fill the non-active columns with a blank string so the table displays cleanly
            for col in included_types:
                if col not in subset:
                    grp[col] = ""
                    
            combos.append(grp)
            
    if combos:
        res = pd.concat(combos, ignore_index=True)
        res['Conv %'] = (res['Purchases'] / res['Visitors'] * 100).round(2)
        res['Rev/Visitor'] = (res['Revenue'] / res['Visitors']).round(2)
        
        # Filter by Traffic Floor and Sort
        final_res = res[res['Visitors'] >= min_visitors].sort_values(metric_map[metric_choice], ascending=False)
        
        # Order the columns so the traits come first, metrics come last
        metrics = ["Visitors", "Purchases", "Revenue", "Conv %", "Rev/Visitor"]
        ordered_cols = included_types + metrics
        
        rename_dict = {"gender": "Gender", "age": "Age", "income": "Income", "region": "Region", "net_worth": "Net Worth", "children": "Children", "marital_status": "Marital Status", "homeowner": "Homeowner", "credit_rating": "Credit Rating"}
        
        if final_res.empty:
            st.warning(f"No combinations met the Traffic Floor minimum of {min_visitors}.")
        else:
            st.dataframe(
                final_res[ordered_cols].rename(columns=rename_dict).style.format({'Conv %': '{:.2f}%', 'Revenue': '${:,.2f}', 'Rev/Visitor': '${:,.2f}'}).background_gradient(subset=['Rev/Visitor', 'Conv %'], cmap='YlGn'), 
                use_container_width=True
            )
elif not included_types:
    st.info("👆 Check the 'Inc' boxes to build your combination matrix.")

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
