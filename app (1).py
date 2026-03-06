import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ================ 1. Page Config & Premium Brand CSS =================
st.set_page_config(page_title="Audience Engine | ICP", page_icon="🎯", layout="wide")

def apply_custom_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
            .stApp { background-color: #FAFAFA; }
            
            h1, h2, h3 { color: #111827 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
            p, span, label { color: #4B5563 !important; }
            
            /* Clean Input Boxes */
            .stTextInput input { border-radius: 8px; border: 1px solid #E5E7EB; padding: 12px 16px; }
            .stTextInput input:focus { border-color: #3B82F6; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); }
            
            /* Metric Cards */
            [data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 20px 24px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
            }
            [data-testid="stMetricLabel"] { color: #6B7280 !important; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; }
            [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700; font-size: 2.2rem; }
            
            /* === LUXURY HTML TABLE STYLING === */
            .premium-table-container {
                width: 100%;
                overflow-x: auto;
                border-radius: 12px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 4px 10px rgba(0,0,0, 0.02);
                background: #FFFFFF;
                margin-bottom: 2rem;
            }
            .premium-table-container table {
                width: 100% !important;
                border-collapse: collapse !important;
                font-family: 'Outfit', sans-serif !important;
                margin-bottom: 0 !important;
            }
            .premium-table-container table thead tr th,
            .premium-table-container table th {
                background-color: #F3F4F6 !important;
                color: #374151 !important;
                font-weight: 700 !important;
                text-align: center !important; 
                padding: 12px 14px !important; 
                border-bottom: 2px solid #E5E7EB !important;
                text-transform: uppercase !important;
                font-size: 0.75rem !important;
                white-space: nowrap !important;
            }
            .premium-table-container table tbody tr td,
            .premium-table-container table td {
                text-align: center !important; 
                padding: 12px 14px !important; 
                border-bottom: 1px solid #F9FAFB !important;
                color: #111827 !important;
                font-size: 0.90rem !important;
                white-space: nowrap !important;
            }
            .premium-table-container table tbody tr td:first-child,
            .premium-table-container table tbody tr th:first-child {
                font-weight: 700 !important;
                text-align: left !important; 
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# Helper function to render perfect HTML tables
def render_premium_table(styler_obj):
    try:
        styler_obj = styler_obj.hide(axis="index")
    except AttributeError:
        styler_obj = styler_obj.hide_index() 
    html = styler_obj.to_html()
    st.markdown(f'<div class="premium-table-container">{html}</div>', unsafe_allow_html=True)

# ================ 2. BigQuery Connection =================
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    # Handle private key formatting issues
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    return bigquery.Client(credentials=service_account.Credentials.from_service_account_info(creds_dict), project=creds_dict["project_id"])

# ================ 3. The Onboarding Flow =================
st.markdown("<h1 style='text-align: center;'>🎯 Build Your Ideal Customer Profile</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 3rem;'>Upload your recent orders to see the demographic DNA of your actual paying customers.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    website_url = st.text_input("🔗 What is your website URL?", placeholder="https://www.yourstore.com")
    if website_url:
        st.success(f"Scanning `{website_url}` for brand assets... (Visual theming coming soon!)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📦 Upload your Order Data** (Required columns: `Date`, `Email`, `Order ID`, `Total`)")
    uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"], label_visibility="collapsed")

# ================ 4. Data Processing & ICP Dashboard =================
if uploaded_file is not None:
    try:
        df_orders = pd.read_csv(uploaded_file)
        
        # 1. Validation Check
        required_cols = ["Date", "Email", "Order ID", "Total"]
        missing_cols = [col for col in required_cols if col not in df_orders.columns]
        
        if missing_cols:
            st.error(f"⚠️ Your CSV is missing: **{', '.join(missing_cols)}**")
        else:
            with st.spinner("Pushing orders to BigQuery and matching pixels... This takes about 5 seconds."):
                client = get_bq_client()
                
                # 2. Push CSV to a Temporary Staging Table in BigQuery
                project_id = dict(st.secrets["gcp_service_account"])["project_id"]
                dataset_id = "final_dashboard"  # Assuming you want to keep temp tables here
                temp_table_id = f"{project_id}.{dataset_id}.streamlit_temp_orders"
                
                # Configure job to overwrite the temp table every time
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                job = client.load_table_from_dataframe(df_orders, temp_table_id, job_config=job_config)
                job.result() # Wait for upload to finish
                
                # 3. Run the Massive Join Query mapped to your exact schema
                query = f"""
                    SELECT 
                        o.Email, 
                        o.`Order ID` as Order_ID, 
                        o.Total,
                        p.GENDER as gender, 
                        p.AGE_RANGE as age, 
                        p.INCOME_RANGE as income, 
                        CASE 
                            WHEN p.PERSONAL_STATE IN ('CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA') THEN 'Northeast'
                            WHEN p.PERSONAL_STATE IN ('IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI') THEN 'Midwest'
                            WHEN p.PERSONAL_STATE IN ('AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'DC') THEN 'South'
                            WHEN p.PERSONAL_STATE IN ('AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY') THEN 'West'
                            ELSE 'Unknown'
                        END as region,
                        p.NET_WORTH as net_worth, 
                        p.CHILDREN as children, 
                        p.MARRIED as marital_status, 
                        p.HOMEOWNER as homeowner, 
                        p.SKIPTRACE_CREDIT_RATING as credit_rating
                    FROM `{temp_table_id}` o
                    LEFT JOIN `xenon-mantis-430216-n4.visitors_raw.all_visitors_combined` p
                    ON LOWER(p.PERSONAL_EMAILS) LIKE CONCAT('%', LOWER(o.Email), '%') 
                       OR LOWER(p.BUSINESS_EMAIL) = LOWER(o.Email)
                """
                
                df_joined = client.query(query).to_dataframe()
                df_joined = df_joined.fillna("Unknown").replace("", "Unknown")
            
            # 4. Build the Dashboard
            st.markdown("<hr style='margin-top: 3rem; margin-bottom: 3rem;'>", unsafe_allow_html=True)
            st.header("🧬 Your Customer DNA")
            
            # Baseline KPIs
            total_buyers = df_joined['Email'].nunique()
            total_rev = df_joined['Total'].sum()
            overall_aov = total_rev / total_buyers if total_buyers > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Matched Buyers", f"{total_buyers:,.0f}")
            m2.metric("Total Attributed Revenue", f"${total_rev:,.2f}")
            m3.metric("Overall Average Order Value", f"${overall_aov:,.2f}")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Demographic Breakdowns
            demographics = [
                ("Gender", "gender"), 
                ("Age", "age"), 
                ("Income", "income"), 
                ("Region", "region"), 
                ("Net Worth", "net_worth"), 
                ("Marital Status", "marital_status"),
                ("Homeowner", "homeowner"),
                ("Credit Rating", "credit_rating")
            ]
            
            # Create two columns to stack tables nicely
            dash_col1, dash_col2 = st.columns(2)
            
            for index, (label, col_name) in enumerate(demographics):
                # Calculate Share of Wallet metrics
                grp = df_joined.groupby(col_name).agg(
                    Buyers=('Email', 'nunique'),
                    Revenue=('Total', 'sum')
                ).reset_index()
                
                # Filter out "Unknowns" for the clean ICP view
                grp = grp[grp[col_name] != "Unknown"]
                
                if not grp.empty:
                    grp['% of Buyers'] = (grp['Buyers'] / grp['Buyers'].sum()) * 100
                    grp['% of Revenue'] = (grp['Revenue'] / grp['Revenue'].sum()) * 100
                    grp['AOV'] = grp['Revenue'] / grp['Buyers']
                    
                    # Sort by Revenue generated
                    grp = grp.sort_values('Revenue', ascending=False)
                    grp = grp.rename(columns={col_name: label})
                    
                    # Formatting
                    format_dict = {'Buyers': '{:,.0f}', '% of Buyers': '{:.1f}%', 'Revenue': '${:,.2f}', '% of Revenue': '{:.1f}%', 'AOV': '${:,.2f}'}
                    
                    # Highlight the highest revenue contributors in green
                    styler = grp.style.format(format_dict).background_gradient(subset=['% of Revenue', '% of Buyers'], cmap="Greens")
                    
                    # Place in alternating columns
                    with dash_col1 if index % 2 == 0 else dash_col2:
                        st.subheader(f"{label} Distribution")
                        render_premium_table(styler)

    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")
