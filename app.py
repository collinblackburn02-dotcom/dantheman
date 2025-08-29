
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import coerce_purchase_series, pick_default_purchase_col, safe_percent, detect_date_col, coerce_datetime

st.set_page_config(page_title="CSV Dashboard", layout="wide")

st.markdown("## 📊 Customer Analytics Dashboard")
st.caption("Upload a CSV, choose your purchase column, apply filters, and explore KPIs, leaderboards (1D) and multi-D rankings (up to 4D). Export filtered data anytime.")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    st.divider()
    st.write("**Optional column mapping**")

@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df

def agg_by_dim(d: pd.DataFrame, dim: str):
    g = d.groupby(dim, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    g["conv_rate"] = (g["purchases"] / g["rows"]).replace([np.inf, -np.inf], np.nan) * 100
    g["conv_rate"] = g["conv_rate"].fillna(0.0)
    g[dim] = g[dim].astype(str).replace("nan","(blank)")
    return g

def plot_by_dim_table_and_bar(d: pd.DataFrame, dim: str, sort_metric: str, min_rows: int, top_n: int):
    g = agg_by_dim(d, dim)
    g = g[g["rows"] >= min_rows]
    if g.empty:
        st.info(f"No groups meet the minimum rows ≥ {min_rows} for {dim}.")
        return
    # Sort DESC for all metrics (Top = highest)
    ascending = False
    g_sorted = g.sort_values(sort_metric, ascending=ascending).copy()
    g_sorted = g_sorted.iloc[:top_n]
    gt = g_sorted.rename(columns={dim: dim, "rows":"Visitors", "purchases":"Purchases", "conv_rate":"Conversion %"})
    gt["Conversion %"] = gt["Conversion %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(gt, use_container_width=True, hide_index=True)
    metric_title = {"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"}[sort_metric]
    y = "conv_rate" if sort_metric == "conv_rate" else sort_metric
    fig = px.bar(g_sorted, x=dim, y=y, text=g_sorted[y].round(2), title=f"Top {top_n} by {metric_title} • {dim}")
    if y == "conv_rate":
        fig.update_yaxes(title="Conversion (%)")
    st.plotly_chart(fig, use_container_width=True)

def multi_dim_ranker(d: pd.DataFrame, dims: list[str], sort_metric_key: str, min_rows: int, top_n: int):
    if len(dims) == 0:
        st.info("Select 1–4 dimensions to build the combination leaderboard.")
        return
    if len(dims) > 4:
        st.warning("Only the first 4 dimensions will be used.")
        dims = dims[:4]
    g = d.groupby(dims, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    g["conv_rate"] = (g["purchases"] / g["rows"]).replace([np.inf, -np.inf], np.nan) * 100
    # Clean NaN labels
    for dim in dims:
        g[dim] = g[dim].astype(str).replace("nan","(blank)")
    # Apply minimum sample threshold
    g = g[g["rows"] >= min_rows]
    if g.empty:
        st.info(f"No combinations meet the minimum rows ≥ {min_rows}.")
        return
    # Sort (DESC for all metrics)
    g_sorted = g.sort_values(sort_metric_key, ascending=False).copy()
    g_top = g_sorted.head(top_n).copy()
    # Prepare display
    display = g_top.rename(columns={"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"})
    display["Conversion %"] = display["Conversion %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)
    # Optional: downloadable CSV with raw numeric conv_rate
    st.download_button(
        "Download ranked combinations (CSV)",
        data=g_sorted.head(top_n).to_csv(index=False).encode("utf-8"),
        file_name="ranked_combinations.csv",
        mime="text/csv"
    )

def get_filter(df: pd.DataFrame, col: str, label: str):
    if col in df.columns:
        opts = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip() != "" ])
        sel = st.multiselect(label, opts, default=[])
        if sel:
            return df[df[col].isin(sel)]
    return df

if uploaded:
    df = load_df(uploaded)
    with st.sidebar:
        default_purchase_col = pick_default_purchase_col(df) or st.selectbox("Purchase Indicator Column (choose)", list(df.columns))
        purchase_col = st.selectbox("Purchase Indicator Column", 
                                    options=[default_purchase_col] + [c for c in df.columns if c != default_purchase_col]) if default_purchase_col else st.selectbox("Purchase Indicator Column", list(df.columns))
        st.caption("Numeric: >0 = purchased. Text: 'yes/true/1/buyer' = purchased.")

        guessed_date = detect_date_col(df)
        date_col = st.selectbox("Date Column (optional)", ["" ] + list(df.columns), index=(list(df.columns).index(guessed_date)+1 if guessed_date in df.columns else 0))

        st.markdown("**Enable filters for these columns (if present):**")
        filter_cols = []
        for c in ["GENDER","AGE_RANGE","INCOME_RANGE","NET_WORTH","HOMEOWNER","SKIPTRACE_CREDIT_RATING","MARRIED","CHILDREN"]:
            if c in df.columns and st.checkbox(c, value=(c in ["GENDER","INCOME_RANGE","SKIPTRACE_CREDIT_RATING","HOMEOWNER"])):
                filter_cols.append(c)

        st.divider()
        st.write("**Chart settings**")
        y_metric_mode = st.radio("Y-axis metric", ["Purchases", "Conversion Rate"], horizontal=True)

    try:
        df["_PURCHASE"] = coerce_purchase_series(df, purchase_col)
    except Exception as e:
        st.error(f"Could not interpret purchase column `{purchase_col}`. Error: {e}")
        st.stop()
    if date_col and date_col in df.columns:
        df["_DATE"] = coerce_datetime(df, date_col)
    else:
        df["_DATE"] = pd.NaT

    # Filters
    with st.expander("🔎 Filters", expanded=True):
        dff = df.copy()
        for c in filter_cols:
            dff = get_filter(dff, c, f"Filter by {c}")
        if not dff["_DATE"].isna().all():
            min_d = pd.to_datetime(dff["_DATE"].min())
            max_d = pd.to_datetime(dff["_DATE"].max())
            if pd.notna(min_d) and pd.notna(max_d):
                start, end = st.date_input("Date range", (min_d.date(), max_d.date()))
                if not isinstance(start, tuple):
                    dff = dff[(dff["_DATE"] >= pd.to_datetime(start)) & (dff["_DATE"] <= pd.to_datetime(end))]

    # KPIs
    total_rows = len(dff)
    total_purch = int(dff["_PURCHASE"].sum())
    conv_rate = safe_percent(total_purch, total_rows)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Rows", f"{total_rows:,}")
    kpi2.metric("Purchases", f"{total_purch:,}")
    kpi3.metric("Conversion Rate", f"{conv_rate:.2f}%")

    st.divider()

    # Leaderboard 1D
    st.subheader("🏆 Leaderboards (1D)")
    dims_available = [c for c in ["INCOME_RANGE","SKIPTRACE_CREDIT_RATING","NET_WORTH","GENDER","HOMEOWNER","AGE_RANGE","MARRIED","CHILDREN"] if c in dff.columns]
    if len(dims_available) == 0:
        st.info("No standard segmentation columns found. Upload a CSV with columns like INCOME_RANGE, GENDER, NET_WORTH, HOMEOWNER, or SKIPTRACE_CREDIT_RATING to enable leaderboards.")
    else:
        left, right = st.columns(2)
        with left:
            dim1 = st.selectbox("Dimension to rank", options=dims_available, index=0)
            sort_metric = st.selectbox("Sort by", options=["Conversion %","Purchases","Visitors"], index=0)
            min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)
            top_n = st.slider("Top N", min_value=3, max_value=100, value=10, step=1)
            sort_key = {"Conversion %":"conv_rate", "Purchases":"purchases", "Visitors":"rows"}[sort_metric]
            plot_by_dim_table_and_bar(dff, dim1, sort_key, min_rows, top_n)

        with right:
            st.markdown("**Multi-D Combinations (up to 4D)**")
            combo_dims = st.multiselect("Choose 1–4 dimensions", options=dims_available, default=[dims_available[0]] if len(dims_available)>0 else [])
            metric2 = st.selectbox("Metric", options=["Conversion %","Purchases","Visitors"], index=0)
            min_rows2 = st.number_input("Minimum Visitors per combination", min_value=1, value=20, step=1, key="min_rows2")
            top_n2 = st.slider("Top N combinations", min_value=3, max_value=200, value=25, step=1, key="topn2")
            sort_key2 = {"Conversion %":"conv_rate", "Purchases":"purchases", "Visitors":"rows"}[metric2]
            multi_dim_ranker(dff, combo_dims, sort_key2, min_rows2, top_n2)

    st.divider()

    # Trend chart
    if not dff["_DATE"].isna().all():
        st.subheader("📈 Trend Over Time")
        ts = dff.copy()
        ts["date_only"] = ts["_DATE"].dt.date
        line = ts.groupby("date_only")["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
        line["conv_rate"] = (line["purchases"] / line["rows"]) * 100
        y = "purchases" if y_metric_mode == "Purchases" else "conv_rate"
        y_title = "Purchases" if y == "purchases" else "Conversion Rate (%)"
        fig = px.line(line, x="date_only", y=y, markers=True, title="Trend Over Time")
        fig.update_layout(yaxis_title=y_title, xaxis_title="Date", margin=dict(l=10,r=10,b=40,t=60))
        st.plotly_chart(fig, use_container_width=True)

    # Table + Export
    st.subheader("📥 Filtered Data")
    st.dataframe(dff, use_container_width=True, height=400)
    st.download_button("Download filtered CSV", data=dff.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_data.csv", mime="text/csv")

else:
    st.info("Upload a CSV file from the sidebar to get started.")
