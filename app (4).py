import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import altair as alt

# ================ 1. CONFIGURATION & THEME =================
PITCH_COMPANY_NAME = "LeadNavigator" 
PITCH_BRAND_COLOR = "#B3845C" 
DEMO_PASSWORD = "leadnavai"

AWS_COLUMN_MAPPER = {
    "GENDER": "gender",
    "MARRIED": "marital_status",
    "AGE_RANGE": "age",
    "INCOME_RANGE": "income",
    "PERSONAL_STATE": "state_raw",
    "PERSONAL_ZIP": "zip_code", 
    "NET_WORTH": "net_worth",
    "SKIPTRACE_CREDIT_RATING": "credit_rating"
}

STATE_TO_REGION = {
    'CT':'Northeast','MA':'Northeast','ME':'Northeast','NH':'Northeast','NJ':'Northeast','NY':'Northeast','PA':'Northeast','RI':'Northeast','VT':'Northeast',
    'IA':'Midwest','IL':'Midwest','IN':'Midwest','KS':'Midwest','MI':'Midwest','MN':'Midwest','MO':'Midwest','ND':'Midwest','NE':'Midwest','OH':'Midwest','SD':'Midwest','WI':'Midwest',
    'AL':'South','AR':'South','DC':'South','DE':'South','FL':'South','GA':'South','KY':'South','LA':'South','MD':'South','MS':'South','NC':'South','OK':'South','SC':'South','TN':'South','TX':'South','VA':'South','WV':'South',
    'AK':'West','AZ':'West','CA':'West','CO':'West','HI':'West','ID':'West','MT':'West','NM':'West','NV':'West','OR':'West','UT':'West','WA':'West','WY':'West'
}

st.set_page_config(page_title=f"{PITCH_COMPANY_NAME} | Audience Engine", page_icon="🧬", layout="wide")

def apply_custom_theme(primary_color):
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
            .stApp {{ background-color: #F9F7F3; }}
            h1, h2, h3 {{ color: #2D2421 !important; font-weight: 600 !important; }}
            div[data-testid="stButton"] button {{ border-radius: 8px; font-weight: 500; padding: 0px 10px !important; }}
            div[data-testid="stButton"] button[kind="primary"] {{ background-color: {primary_color} !important; color: #FFFFFF !important; border: none; }}
            div[data-testid="stButton"] button[kind="secondary"] {{ background-color: #FFFFFF; color: #2D2421; border: 1px solid #E2D7C8; }}
            [data-testid="stMetric"] {{ background-color: #FFFFFF; border: 1px solid #E2D7C8; border-radius: 12px; padding: 20px; text-align: center; }}
            [data-testid="stMetricDelta"] {{ color: #09AB3B !important; }}
            [data-testid="stMetricDelta"] svg {{ display: none; }} 
            .premium-table-container {{ border-radius: 12px; border: 1px solid #E2D7C8; background: #FFFFFF; overflow: hidden; margin-top: 1rem; }}
            .premium-table-container table {{ width: 100% !important; border-collapse: collapse !important; }}
            .premium-table-container th {{ background-color: #F2EBE1 !important; color: #3A2A26 !important; font-weight: 700 !important; text-align: center !important; padding: 12px !important; border-bottom: 2px solid #D5C6B3 !important; text-transform: uppercase !important; font-size: 0.75rem !important; }}
            .premium-table-container td {{ text-align: center !important; padding: 12px !important; border-bottom: 1px solid #F0EAD6 !important; font-size: 0.85rem !important; }}
            .premium-table-container td:first-child {{ font-weight: 700 !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme(PITCH_BRAND_COLOR)
custom_light_green = mcolors.LinearSegmentedColormap.from_list("custom_green", ["#F9F7F3", "#D1E5D1", "#6EAB6E"])

def render_premium_table(styler_obj):
    st.markdown(f'<div class="premium-table-container">{styler_obj.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# ================ 2. DATA ENGINE =================
@st.cache_data(ttl=3600) 
def load_master_graph():
    aws_keys = {"key": st.secrets["aws"]["access_key"], "secret": st.secrets["aws"]["secret_key"], "client_kwargs": {"region_name": "us-east-2"}}
    files = ["master_data.csv", "visitor_data_2.csv"] 
    dataframes = []
    try:
        for f in files:
            path = f"s3://leadnav-demo-data/{f}"
            temp_df = pd.read_csv(path, storage_options=aws_keys, low_memory=False, encoding='latin1', on_bad_lines='skip')
            temp_df.columns = [c.upper().strip() for c in temp_df.columns]
            e_col = 'PERSONAL_EMAILS' if 'PERSONAL_EMAILS' in temp_df.columns else temp_df.columns[0]
            temp_df = temp_df.rename(columns={e_col: 'email_match'})
            dataframes.append(temp_df)
            
        df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
        df = df.rename(columns=AWS_COLUMN_MAPPER)
        df.columns = [c.lower() for c in df.columns]
        
        if 'state_raw' in df.columns: df['region'] = df['state_raw'].str.strip().str.upper().map(STATE_TO_REGION).fillna('Unknown')
        if 'gender' in df.columns: df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')
        if 'marital_status' in df.columns: df['marital_status'] = df['marital_status'].map({'Y': 'Married', 'N': 'Single'}).fillna('Unknown')
        
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str).str.replace(r'\.0$', '', regex=True)
            df.loc[df['zip_code'].str.lower().isin(['nan', 'none', '', 'unknown']), 'zip_code'] = None
            df['zip_code'] = df['zip_code'].str.zfill(5)
        
        df['email_match'] = df['email_match'].astype(str).str.lower().str.replace(r'[^a-z0-9@._-]', '', regex=True).str.split(',')
        df = df.explode('email_match').reset_index(drop=True)
        return df.drop_duplicates(subset=['email_match']).reset_index(drop=True)
    except Exception as e:
        st.error(f"🚨 AWS Error: {e}"); st.stop()

# ================ 3. DASHBOARD =================
if "app_state" not in st.session_state: st.session_state.app_state = "onboarding"
if "is_unlocked" not in st.session_state: st.session_state.is_unlocked = False

if st.session_state.app_state == "onboarding":
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-top: 50px;'>🎯 Audience Engine</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])
        if uploaded_file:
            df_orders = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
            df_orders = df_orders.rename(columns={'Email': 'email_match', 'Name': 'Order ID', 'Total': 'revenue_raw'})
            with st.spinner("Executing LeadNavigator Resolution..."):
                df_master = load_master_graph()
                df_orders['email_match'] = df_orders['email_match'].astype(str).str.lower().str.strip()
                df_joined = pd.merge(df_orders, df_master, on='email_match', how='inner').reset_index(drop=True)
                if not df_joined.empty:
                    st.session_state.df_icp = df_joined
                    st.session_state.app_state = "dashboard"
                    st.rerun()

elif st.session_state.app_state == "dashboard":
    if not st.session_state.is_unlocked:
        @st.dialog("🔒 Secure Access Required")
        def login_modal():
            st.write("Please enter the password to view the complete Customer DNA profile.")
            pwd = st.text_input("Password", type="password")
            if st.button("Unlock Dashboard", use_container_width=True, kind="primary"):
                if pwd == DEMO_PASSWORD:
                    st.session_state.is_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect Password")
        
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("🔑 Unlock Full Profile", kind="primary"): login_modal()
        if c2.button("🔄 New Analysis"): st.session_state.app_state = "onboarding"; st.rerun()
    else:
        c1, _ = st.columns([1, 5])
        if c1.button("🔄 New Analysis"): st.session_state.app_state = "onboarding"; st.session_state.is_unlocked = False; st.rerun()

    full_df = st.session_state.df_icp
    df_p = full_df if st.session_state.is_unlocked else full_df.head(100).copy()
    df_p['revenue_raw'] = pd.to_numeric(df_p['revenue_raw'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)
    
    # 1. MACRO METRICS
    m1, m2 = st.columns(2)
    m1.metric("Resolved Profiles", f"{df_p['Order ID'].nunique():,.0f}")
    m2.metric("Attributed Sales", f"${df_p['revenue_raw'].sum():,.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

    # 2. TOP PERFORMING DEMOGRAPHICS
    st.markdown("### 🏆 Top Performing Demographics")
    total_rev = df_p['revenue_raw'].sum()
    summary_vars = [("Gender", "gender"), ("Age", "age"), ("Marital Status", "marital_status"), ("Region", "region"), ("State", "state_raw"), ("Zip Code", "zip_code"), ("Credit Rating", "credit_rating")]
    summary_cols = st.columns(len(summary_vars))
    for idx, (label, col_key) in enumerate(summary_vars):
        if col_key in df_p.columns:
            temp = df_p[~df_p[col_key].astype(str).str.lower().isin(['unknown', 'nan', 'u', 'none', '00nan'])]
            if not temp.empty:
                rev_series = temp.groupby(col_key)['revenue_raw'].sum()
                winner = rev_series.idxmax()
                rev_pct = (rev_series.max() / total_rev * 100) if total_rev > 0 else 0
                summary_cols[idx].metric(label, winner, f"{rev_pct:.1f}% of Revenue")

    st.markdown("<hr>", unsafe_allow_html=True)

    # 3. SINGLE VARIABLE DEEP DIVE
    st.markdown("### 🔍 Single Variable Deep Dive")
    configs = [("Gender", "gender"), ("Age", "age"), ("Location", "location"), ("Marital Status", "marital_status"), ("Credit Rating", "credit_rating")]
    if "active_var" not in st.session_state: st.session_state.active_var = "Gender"
    if "active_loc_level" not in st.session_state: st.session_state.active_loc_level = "Region"
    
    var_cols = st.columns(len(configs))
    for i, (label, col_name) in enumerate(configs):
        if var_cols[i].button(label, key=f"btn_{label}", type="primary" if st.session_state.active_var == label else "secondary", use_container_width=True):
            st.session_state.active_var = label
            st.rerun()

    if st.session_state.active_var == "Location":
        st.markdown("<br>", unsafe_allow_html=True)
        l_col1, l_col2, l_col3, _ = st.columns([1, 1, 1, 5])
        if l_col1.button("Region", type="primary" if st.session_state.active_loc_level == "Region" else "secondary"): st.session_state.active_loc_level = "Region"; st.rerun()
        if l_col2.button("State", type="primary" if st.session_state.active_loc_level == "State" else "secondary"): st.session_state.active_loc_level = "State"; st.rerun()
        if l_col3.button("Zip Code", type="primary" if st.session_state.active_loc_level == "Zip Code" else "secondary"): st.session_state.active_loc_level = "Zip Code"; st.rerun()
        loc_map = {"Region": "region", "State": "state_raw", "Zip Code": "zip_code"}
        active_col = loc_map[st.session_state.active_loc_level]
        display_label = st.session_state.active_loc_level
    else:
        active_col = dict(configs)[st.session_state.active_var]
        display_label = st.session_state.active_var

    if active_col in df_p.columns:
        df_clean = df_p[~df_p[active_col].astype(str).str.lower().isin(['unknown', 'nan', 'u', 'none', '00nan'])]
        df_p_grp = df_clean.groupby(active_col).agg(Purchasers=('Order ID', 'nunique'), Revenue=('revenue_raw', 'sum')).reset_index()
        if not df_p_grp.empty:
            df_p_grp['% of Buyers'] = (df_p_grp['Purchasers'] / df_p_grp['Purchasers'].sum()) * 100
            df_p_grp['Rev / Purchaser'] = (df_p_grp['Revenue'] / df_p_grp['Purchasers'])
            display_df = df_p_grp.rename(columns={active_col: display_label.upper()}).sort_values('Revenue', ascending=False)
            render_premium_table(display_df.style.format({'Purchasers': '{:,.0f}', 'Revenue': '${:,.2f}', '% of Buyers': '{:.1f}%', 'Rev / Purchaser': '${:,.2f}'}).background_gradient(subset=['Rev / Purchaser', '% of Buyers'], cmap=custom_light_green))

            # 🚨 REVENUE HEATMAP EXPANDER
            if st.session_state.active_var == "Location":
                with st.expander(f"🗺️ View {display_label} Revenue Heatmap", expanded=True):
                    if st.session_state.active_loc_level in ["State", "Region"]:
                        # Geographic Heatmap (State/Region level)
                        map_chart = alt.Chart(display_df).mark_bar().encode(
                            x=alt.X('Revenue:Q', title="Total Revenue"),
                            y=alt.Y(f'{display_label.upper()}:N', sort='-x', title=display_label),
                            color=alt.Color('Revenue:Q', scale=alt.Scale(scheme='greens'), legend=None),
                            tooltip=[display_label.upper(), alt.Tooltip('Revenue:Q', format='$,.2f')]
                        ).properties(height=350, title=f"Revenue Distribution by {display_label}")
                        st.altair_chart(map_chart, use_container_width=True)
                    else:
                        # Zip Code Concentration Chart
                        zip_chart = alt.Chart(display_df.head(20)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                            x=alt.X(f'{display_label.upper()}:N', sort='-y', title="Zip Code"),
                            y=alt.Y('Revenue:Q', title="Total Revenue"),
                            color=alt.Color('Revenue:Q', scale=alt.Scale(scheme='greens'), legend=None),
                            tooltip=[display_label.upper(), alt.Tooltip('Revenue:Q', format='$,.2f')]
                        ).properties(height=350, title="Top 20 High-Revenue Zip Codes")
                        st.altair_chart(zip_chart, use_container_width=True)
            display_df = df_p_grp.rename(columns={active_col: display_label.upper()}).sort_values('Revenue', ascending=False)
            styler = display_df.style.format({'Purchasers': '{:,.0f}', 'Revenue': '${:,.2f}', '% of Buyers': '{:.1f}%', 'Rev / Purchaser': '${:,.2f}'}).background_gradient(subset=['Rev / Purchaser', '% of Buyers'], cmap=custom_light_green)
            render_premium_table(styler)
