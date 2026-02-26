import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ Brand palette & CSS =================
BRAND = {"bg": "#F5F0E6", "fg": "#3A2A26", "accent": "#6E4F3A", "card": "#FFF9F0"}

def inject_css():
    st.markdown(f"""
        <style>
            .stApp {{ background: var(--bg); color: var(--fg); }}
            .attr-card {{
                background-color: var(--card);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(58,42,38,0.1);
                margin-bottom: 20px;
            }}
            .attr-title {{ font-weight: 800; font-size: 1.1rem; color: var(--fg); margin-bottom: 10px; }}
            .stDataFrame {{ border-radius: 12px; background: var(--card); }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Insights", layout="wide")
inject_css()

# ================ Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

@st.cache_data(ttl=600)
def load_data():
    return get_bq_client().query("SELECT * FROM `final_dashboard.demographic_leaderboard`").to_dataframe()

df = load_data()

# ================ Sidebar =================
with st.sidebar:
    st.header("Leaderboard Settings")
    metric_choice = st.radio("Sort by:", ["Conversion %", "Purchasers", "Visitors"])
    min_visitors = st.number_input("Min Visitors per segment", value=20)
    st.markdown("---")
    max_depth = st.slider("Combination Depth", 1, 4, 2)

# ================ Main UI: Attribute Cards =================
st.title("🪷 Ranked Insights")
st.markdown("Build and filter your target personas using the cards below.")

col1, col2, col3 = st.columns(3)

# Map internal column names to pretty labels
attr_config = {
    "gender": "Gender",
    "age": "Age Range",
    "income": "Income Range",
    "tenure": "Homeowner Status",
    "net_worth": "Net Worth",
    "state": "State",
    "children": "Children"
}

filters = {}
included_cols = []

# Create the Grid
for i, (col_name, label) in enumerate(attr_config.items()):
    if col_name not in df.columns: continue
    
    target_col = [col1, col2, col3][i % 3]
    with target_col:
        st.markdown(f'<div class="attr-card">', unsafe_allow_html=True)
        c_title, c_check = st.columns([3, 1])
        c_title.markdown(f'<p class="attr-title">{label}</p>', unsafe_allow_html=True)
        
        # Checkbox to include in Persona string
        is_inc = c_check.checkbox("Inc", value=(i < 2), key=f"inc_{col_name}")
        
        if is_inc:
            included_cols.append(col_name)
            # Multi-select for filtering specific values
            options = sorted([str(x) for x in df[col_name].dropna().unique().tolist()])
            selected = st.multiselect(f"Filter {label}", options, key=f"sel_{col_name}", label_visibility="collapsed")
            if selected:
                filters[col_name] = selected
        st.markdown('</div>', unsafe_allow_html=True)

# ================ Dynamic Aggregation =================
if not included_cols:
    st.info("Check 'Inc' on the cards above to start building segments.")
else:
    # 1. Filter raw data based on card selections
    dff = df.copy()
    for col, vals in filters.items():
        dff = dff[dff[col].isin(vals)]

    # 2. Apply Combination Depth Limit
    active_cols = included_cols[:max_depth]

    # 3. Create Persona String
    def make_cluster(row):
        parts = [str(row[c]) for c in active_cols if pd.notna(row[c])]
        return " + ".join(parts) if parts else "General Traffic"

    dff['Persona'] = dff.apply(make_cluster, axis=1)

    # 4. Sum up the unique visitor/purchaser counts
    leaderboard = dff.groupby('Persona').agg({
        'total_visitors': 'sum',
        'total_purchasers': 'sum'
    }).reset_index()

    # 5. Calculate Final Metrics
    leaderboard['Conversion %'] = (leaderboard['total_purchasers'] / leaderboard['total_visitors'] * 100).round(2)
    leaderboard = leaderboard[leaderboard['total_visitors'] >= min_visitors]

    # 6. Sorting
    metric_map = {"Conversion %": "Conversion %", "Purchasers": "total_purchasers", "Visitors": "total_visitors"}
    leaderboard = leaderboard.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
    leaderboard.index += 1

    # 7. Display
    st.subheader("Leaderboard")
    st.dataframe(
        leaderboard.rename(columns={'total_visitors': 'Visitors', 'total_purchasers': 'Purchasers'})
        .style.format({'Conversion %': '{:.2f}%'})
        .background_gradient(subset=['Conversion %'], cmap='YlGn'),
        use_container_width=True
    )
