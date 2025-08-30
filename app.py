import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
from utils import resolve_col

st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")

# Logo + Title
c1, c2 = st.columns([0.12, 0.88])
with c1:
    try:
        st.image("logo.png", use_column_width=True)
    except Exception:
        pass
with c2:
    st.markdown("<h1 style='margin-bottom:0'>Heavenly Health — Customer Insights</h1>", unsafe_allow_html=True)
    st.caption("Fast, ranked customer segments powered by DuckDB.")

with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    metric_choice = st.radio("Sort metric", ["Conversion %","Purchases","Visitors","Revenue / Visitor"], index=0)
    max_depth = st.slider("Max combo depth", 1, 4, 2, 1)
    top_n = st.slider("Top N rows", 10, 1000, 50, 10)

@st.cache_data
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

if uploaded:
    df = load_df(uploaded)
    email_col = resolve_col(df,"EMAIL")
    purchase_col = resolve_col(df,"PURCHASE")
    state_col = resolve_col(df,"PERSONAL_STATE")
    sku_col = resolve_col(df,"MOST_RECENT_SKU")
    revenue_col = resolve_col(df,"REVENUE")

    # Purchase flag
    df["_PURCHASE"] = df[purchase_col].apply(lambda x: 1 if str(x).lower() in ["1","true","yes","y","buyer","purchased"] else 0)

    if revenue_col:
        df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
    else:
        df["_REVENUE"] = 0

    # Fixed SKU order
    fixed_skus = ["ECO","FIR1","FIR2","RL2","TRA2","COM2","OUT2","RL500MID-W","RL900PRO-B","RL900PRO-W"]

    con = duckdb.connect()
    con.register("t", df)

    sku_sums = ", ".join([f"SUM(CASE WHEN {sku_col}='{s}' AND _PURCHASE=1 THEN 1 ELSE 0 END) AS \"SKU:{s}\"" for s in fixed_skus])

    sql = f"""
    SELECT
      COUNT(*) AS Visitors,
      SUM(_PURCHASE) AS Purchases,
      100.0 * SUM(_PURCHASE) / NULLIF(COUNT(*),0) AS conv_rate,
      SUM(_REVENUE) AS revenue,
      1.0 * SUM(_REVENUE) / NULLIF(COUNT(*),0) AS rpv,
      {sku_sums}
    FROM t
    """
    res = con.execute(sql).fetchdf()

    res.insert(0,"Rank",np.arange(1,len(res)+1))
    res["Conversion %"] = res["conv_rate"].map(lambda x:f"{x:.2f}%")

    # Show table
    st.subheader("🏆 Ranked Conversion Table")
    st.dataframe(res, use_container_width=True, hide_index=True)

    # Map by state
    if state_col:
        st.subheader("🗺️ State Map")
        agg = df.groupby(state_col).agg(Visitors=(email_col,"count"),Purchases=("_PURCHASE","sum"),Revenue=("_REVENUE","sum")).reset_index()
        agg["conv_rate"] = 100*agg["Purchases"]/agg["Visitors"]
        agg["rpv"] = agg["Revenue"]/agg["Visitors"]
        color = {"Conversion %":"conv_rate","Purchases":"Purchases","Visitors":"Visitors","Revenue / Visitor":"rpv"}[metric_choice]
        fig = px.choropleth(agg, locations=state_col, locationmode="USA-states", color=color, scope="usa",
                            color_continuous_scale="YlOrBr")
        st.plotly_chart(fig,use_container_width=True)
