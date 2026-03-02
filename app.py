import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os

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
            
            /* Custom styling for our new toggle buttons */
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
    df = client.query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()
    return df.fillna("")

df_master = load_data()

# ================ Sidebar & Global Controls =================
with st.sidebar:
    st.header("Global Controls")
    metric_choice = st.radio("Primary Metric", ["Conv %", "Purchases", "Visitors"])
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

# ================ 2. Logic: Resolve Combinations =================
active_types = []
if included_types:
    inc_set = set(included_types)
    active_types.extend(included_types) 

    if "gender" in inc_set and "age" in inc_set: active_types.append("gender_age")
    if "gender" in inc_set and "income" in inc_set: active_types.append("gender_income")
    if "gender" in inc_set and "state" in inc_set: active_types.append("gender_state")
    if "gender" in inc_set and "net_worth" in inc_set: active_types.append("gender_nw")
    if "gender" in inc_set and "children" in inc_set: active_types.append("gender_children")
    if "age" in inc_set and "income" in inc_set: active_types.append("age_income")
    if "age" in inc_set and "net_worth" in inc_set: active_types.append("age_nw")
    if "state" in inc_set and "income" in inc_set: active_types.append("state_income")
    if "income" in inc_set and "net_worth" in inc_set: active_types.append("income_nw")
    if "state" in inc_set and "net_worth" in inc_set: active_types.append("state_nw")
    if "state" in inc_set and "children" in inc_set: active_types.append("state_children")
    if "net_worth" in inc_set and "children" in inc_set: active_types.append("nw_children")

    if "gender" in inc_set and "credit_rating" in inc_set: active_types.append("gender_credit")
    if "age" in inc_set and "credit_rating" in inc_set: active_types.append("age_credit")
    if "income" in inc_set and "credit_rating" in inc_set: active_types.append("income_credit")
    if "state" in inc_set and "credit_rating" in inc_set: active_types.append("state_credit")
    if "net_worth" in inc_set and "credit_rating" in inc_set: active_types.append("nw_credit")
    if "children" in inc_set and "credit_rating" in inc_set: active_types.append("children_credit")

    if {"gender", "age", "income"}.issubset(inc_set): active_types.append("gender_age_income")
    if {"gender", "age", "state"}.issubset(inc_set): active_types.append("gender_age_state")
    if {"gender", "income", "state"}.issubset(inc_set): active_types.append("gender_income_state")
    if {"gender", "income", "net_worth"}.issubset(inc_set): active_types.append("gender_income_nw")
    if {"gender", "age", "children"}.issubset(inc_set): active_types.append("gender_age_children")
    if {"age", "income", "net_worth"}.issubset(inc_set): active_types.append("age_income_nw")
    if {"state", "income", "net_worth"}.issubset(inc_set): active_types.append("state_income_nw")
    if {"gender", "state", "net_worth"}.issubset(inc_set): active_types.append("gender_state_nw")
    if {"state", "net_worth", "children"}.issubset(inc_set): active_types.append("state_nw_children")
    
    if len(inc_set) == 6: active_types.append("all_6")
    if len(inc_set) == 7: active_types.append("all_7")

# ================ 3. Main Combination Display =================
dff = df_master[df_master['cluster_type'].isin(active_types)].copy()

for col, val in selected_filters.items():
    dff = dff[dff[col] == val]

metric_map = {"Conv %": "Conv %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}

if not dff.empty:
    dff['Conv %'] = (dff['total_purchasers'] / dff['total_visitors'] * 100).round(2)
    dff = dff[dff['total_visitors'] >= min_visitors]

if dff.empty:
    st.warning("No data found for this combination, or it didn't meet the Traffic Floor minimum.")
else:
    sku_cols = []
    if 'sku_string' in dff.columns:
        parsed_skus = dff['sku_string'].apply(
            lambda x: dict(item.split("::") for item in str(x).split("~~") if "::" in item) if pd.notna(x) and x != "" else {}
        )
        sku_df = pd.DataFrame(parsed_skus.tolist(), index=dff.index).fillna(0).astype(int)
        dff = pd.concat([dff, sku_df], axis=1)
        sku_cols = sorted(list(sku_df.columns))

    dff = dff.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    dff.index += 1

    label_to_col = {"Gender": "gender", "Age Range": "age", "Income Range": "income", 
                    "State": "state", "Net Worth": "net_worth", "Children": "children",
                    "Credit Rating": "credit_rating"}
    
    final_display_cols = []
    for label in ["Gender", "Age Range", "Income Range", "State", "Net Worth", "Children", "Credit Rating"]:
        internal_name = label_to_col.get(label)
        if internal_name in included_types:
            final_display_cols.append(internal_name)
    
    final_display_cols += ["total_visitors", "total_purchasers", "Conv %"] + sku_cols
    
    st.dataframe(
        dff[final_display_cols].rename(columns={
            "gender": "Gender", "age": "Age", "income": "Income", 
            "state": "State", "net_worth": "Net Worth", "children": "Children", "credit_rating": "Credit Rating",
            "total_visitors": "Visitors", "total_purchasers": "Purchases"
        }).style.format({'Conv %': '{:.2f}%'}).background_gradient(subset=['Conv %'], cmap='YlGn'),
        use_container_width=True
    )

# ================ 4. NEW: Single Variable Toggle Bar =================
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

# 1. Initialize the memory so the app knows which button is active
if "active_single_var" not in st.session_state:
    st.session_state.active_single_var = "Gender"

# 2. Draw a beautiful row of 7 columns for our buttons
var_cols = st.columns(len(single_var_options))

# 3. Create the buttons
for i, label in enumerate(single_var_options.keys()):
    # If this button is the active one, mark it "primary" (bold/brown). Otherwise "secondary" (white).
    btn_type = "primary" if st.session_state.active_single_var == label else "secondary"
    
    # When clicked, update the memory and instantly refresh the page
    if var_cols[i].button(label, key=f"btn_{label}", type=btn_type, use_container_width=True):
        st.session_state.active_single_var = label
        st.rerun()

# 4. Grab the data for whichever button is currently bolded
selected_single_label = st.session_state.active_single_var
selected_single_col = single_var_options[selected_single_label]

df_single = df_master[df_master['cluster_type'] == selected_single_col].copy()

if not df_single.empty:
    df_single['Conv %'] = (df_single['total_purchasers'] / df_single['total_visitors'] * 100).round(2)
    df_single = df_single[df_single['total_visitors'] >= min_visitors]
    
    if df_single.empty:
        st.info(f"No groups within **{selected_single_label}** met the Traffic Floor minimum of {min_visitors}.")
    else:
        sku_cols_single = []
        if 'sku_string' in df_single.columns:
            parsed_skus_single = df_single['sku_string'].apply(
                lambda x: dict(item.split("::") for item in str(x).split("~~") if "::" in item) if pd.notna(x) and x != "" else {}
            )
            sku_df_single = pd.DataFrame(parsed_skus_single.tolist(), index=df_single.index).fillna(0).astype(int)
            df_single = pd.concat([df_single, sku_df_single], axis=1)
            sku_cols_single = sorted(list(sku_df_single.columns))
            
        df_single = df_single.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
        df_single.index += 1
        
        display_cols = [selected_single_col, "total_visitors", "total_purchasers", "Conv %"] + sku_cols_single
        
        st.dataframe(
            df_single[display_cols].rename(columns={
                selected_single_col: selected_single_label,
                "total_visitors": "Visitors", 
                "total_purchasers": "Purchases"
            }).style.format({'Conv %': '{:.2f}%'}).background_gradient(subset=['Conv %'], cmap='YlGn'),
            use_container_width=True
        )
else:
    st.info("No data available for this variable.")

# ================ 5. NEW: AI Data Agent (Gemini) =================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🤖 Heavenly AI Data Agent")

if "GEMINI_API_KEY" not in st.secrets:
    st.warning("⚠️ **API Key Missing:** Please add your `GEMINI_API_KEY` to your Streamlit secrets to wake up the AI agent.")
else:
    from pandasai import SmartDataframe
    from pandasai.llm import GoogleGemini
    
    # 1. Boot up the LLM (That's me!)
    llm = GoogleGemini(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. Turn your raw data into a "Smart" dataframe that can execute code
    sdf = SmartDataframe(df_master, config={"llm": llm})

    # 3. Initialize Streamlit's built-in chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 4. Draw the previous chat messages on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. The Chat Input Box
    if prompt := st.chat_input("Ask me anything about your audience data..."):
        
        # Display the user's question instantly
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate the AI's response
        with st.chat_message("assistant"):
            with st.spinner("Crunching the numbers..."):
                try:
                    # The AI writes the code, runs it, and generates the answer
                    response = sdf.chat(prompt)
                    st.markdown(response)
                    # Save the answer to memory
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"I hit a snag trying to calculate that: {e}")
