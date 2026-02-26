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
            .attr-title {{ font-weight: 800; font-size: 1.1rem; color: var(--fg); }}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Heavenly Insights", layout="wide")
inject_css()

# ================ Data Connection =================
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

# ================ Sidebar (Metric Only) =================
with st.sidebar:
    st.image("logo.png", width=150)
    metric_choice = st.radio("Sort metric", ["Conversion %", "Purchases", "Visitors"])
    min_visitors = st.number_input("Minimum Visitors", value=10)

# ================ Main Layout: Attribute Grid =================
st.title("Ranked Insights")
st.caption("Fast, ranked customer segments.")

st.markdown("### Attributes")
col1, col2, col3 = st.columns(3)

# Mapping our BigQuery columns to your UI labels
attr_config = {
    "age": "Age Range",
    "income": "Income Range",
    "net_worth": "Net Worth",
    "gender": "Gender",
    "tenure": "Homeowner",
    "children": "Children"
}

filters = {}
active_cols = []

# Generate the grid of cards
for i, (col_name, label) in enumerate(attr_config.items()):
    target_col = [col1, col2, col3][i % 3]
    with target_col:
        st.markdown(f'<div class="attr-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1: st.markdown(f'<p class="attr-title">{label}</p>', unsafe_allow_html=True)
        with c2: include = st.checkbox("Include", value=True, key=f"inc_{col_name}")
        
        if include:
            active_cols.append(col_name)
            options = sorted(df[col_name].dropna().unique().tolist())
            selected = st.multiselect("Choose options", options, key=f"sel_{col_name}", label_visibility="collapsed")
            if selected:
                filters[col_name] = selected
        st.markdown('</div>', unsafe_allow_html=True)

# ================ Dynamic Ranking Logic =================
# Apply multi-select filters
dff = df.copy()
for col, vals in filters.items():
    dff = dff[dff[col].isin(vals)]

# Create the "Persona Cluster" string based on what is 'Included'
def make_cluster(row):
    parts = [str(row[c]) for c in active_cols if pd.notna(row[c])]
    return " + ".join(parts) if parts else "All Traffic"

dff['Persona Cluster'] = dff.apply(make_cluster, axis=1)

# Group and Rank
final = dff.groupby('Persona Cluster').agg({
    'total_visitors': 'sum',
    'total_purchasers': 'sum'
}).reset_index()

final['Conversion %'] = (final['total_purchasers'] / final['total_visitors'] * 100).round(2)
final = final[final['total_visitors'] >= min_visitors]

# Display Table
metric_map = {"Conversion %": "Conversion %", "Purchases": "total_purchasers", "Visitors": "total_visitors"}
final = final.sort_values(metric_map[metric_choice], ascending=False).reset_index(drop=True)
final.index += 1

st.dataframe(
    final.rename(columns={'total_visitors': 'Visitors', 'total_purchasers': 'Purchasers'})
    .style.format({'Conversion %': '{:.2f}%'})
    .background_gradient(subset=['Conversion %'], cmap='YlGn'),
    use_container_width=True
)
