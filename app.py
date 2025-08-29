
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import coerce_purchase_series, pick_default_purchase_col, safe_percent, detect_date_col, coerce_datetime

st.set_page_config(page_title="CSV Dashboard", layout="wide")

st.markdown("## 📊 Customer Analytics Dashboard")
st.caption("Upload a CSV, choose your purchase column, apply filters, and explore KPIs, leaderboards, and charts. Export filtered data anytime.")

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
    # Clean display values for NaNs
    g[dim] = g[dim].astype(str).replace("nan","(blank)")
    return g

def plot_by_dim_table_and_bar(d: pd.DataFrame, dim: str, sort_metric: str, min_rows: int, top_n: int):
    g = agg_by_dim(d, dim)
    # Filter by minimum sample size
    g = g[g["rows"] >= min_rows]
    if g.empty:
        st.info(f"No groups meet the minimum rows ≥ {min_rows} for {dim}.")
        return
    # Sort & Top-N
    g_sorted = g.sort_values(sort_metric, ascending=(sort_metric != "conv_rate")).copy()
    g_sorted = g_sorted.iloc[:top_n]
    # Format table
    gt = g_sorted.rename(columns={dim: dim, "rows":"Visitors", "purchases":"Purchases", "conv_rate":"Conversion %"})
    gt["Conversion %"] = gt["Conversion %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(gt, use_container_width=True, hide_index=True)
    # Bar chart
    metric_title = {"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"}[sort_metric]
    y = "conv_rate" if sort_metric == "conv_rate" else sort_metric
    fig = px.bar(g_sorted, x=dim, y=y, text=g_sorted[y].round(2), title=f"Top {top_n} by {metric_title} • {dim}")
    if y == "conv_rate":
        fig.update_yaxes(title="Conversion (%)")
    st.plotly_chart(fig, use_container_width=True)

def two_dim_heatmap(d: pd.DataFrame, dim_x: str, dim_y: str, metric: str, min_rows: int):
    g = d.groupby([dim_x, dim_y], dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    g["conv_rate"] = (g["purchases"] / g["rows"]).replace([np.inf, -np.inf], np.nan) * 100
    g = g.fillna({dim_x:"(blank)", dim_y:"(blank)", "conv_rate":0.0})
    # filter for min rows
    g = g[g["rows"] >= min_rows]
    if g.empty:
        st.info(f"No segment pairs meet the minimum rows ≥ {min_rows}.")
        return
    value_col = {"Visitors":"rows", "Purchases":"purchases", "Conversion %":"conv_rate"}[metric]
    fig = px.density_heatmap(g, x=dim_x, y=dim_y, z=value_col, color_continuous_scale="Blues", title=f"{metric} Heatmap by {dim_x} × {dim_y}")
    st.plotly_chart(fig, use_container_width=True)
    # Show table too
    show = g.copy()
    show = show.rename(columns={"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"})
    show["Conversion %"] = show["Conversion %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(show, use_container_width=True, hide_index=True)

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

    # Coerce purchase/date
    try:
        df["_PURCHASE"] = coerce_purchase_series(df, purchase_col)
    except Exception as e:
        st.error(f"Could not interpret purchase column `{purchase_col}`. Error: {e}")
        st.stop()
    if date_col and date_col in df.columns:
        df["_DATE"] = coerce_datetime(df, date_col)
    else:
        df["_DATE"] = pd.NaT

    # Filters panel
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

    # ===== Leaderboards =====
    st.subheader("🏆 Leaderboards (Ranking)")
    st.caption("Rank segments by Conversion %, Purchases, or Visitors with Top‑N and minimum sample size controls.")

    dims_available = [c for c in ["INCOME_RANGE","SKIPTRACE_CREDIT_RATING","NET_WORTH","GENDER","HOMEOWNER","AGE_RANGE","MARRIED","CHILDREN"] if c in dff.columns]
    if len(dims_available) == 0:
        st.info("No standard segmentation columns found. Upload a CSV with columns like INCOME_RANGE, GENDER, NET_WORTH, HOMEOWNER, or SKIPTRACE_CREDIT_RATING to enable leaderboards.")
    else:
        left, right = st.columns(2)

        with left:
            dim1 = st.selectbox("Dimension to rank", options=dims_available, index=0)
            sort_metric = st.selectbox("Sort by", options=["Conversion %","Purchases","Visitors"], index=0)
            min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)
            top_n = st.slider("Top N", min_value=3, max_value=50, value=10, step=1)
            sort_key = {"Conversion %":"conv_rate", "Purchases":"purchases", "Visitors":"rows"}[sort_metric]
            plot_by_dim_table_and_bar(dff, dim1, sort_key, min_rows, top_n)

        with right:
            st.markdown("**2‑D Segment Heatmap (optional)**")
            dims2 = ["(none)"] + dims_available
            dim_x = st.selectbox("X Dimension", options=dims2, index=1 if len(dims2) > 1 else 0)
            dim_y = st.selectbox("Y Dimension", options=dims2, index=2 if len(dims2) > 2 else 0)
            metric2 = st.selectbox("Metric", options=["Conversion %","Purchases","Visitors"], index=0)
            min_rows2 = st.number_input("Minimum Visitors per pair", min_value=1, value=20, step=1, key="min_rows2")
            if dim_x != "(none)" and dim_y != "(none)" and dim_x != dim_y:
                two_dim_heatmap(dff, dim_x, dim_y, metric2, min_rows2)
            else:
                st.info("Pick two different dimensions to render a heatmap.")

    st.divider()

    # ===== Trend chart =====
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

    # ===== Table + Export =====
    st.subheader("📥 Filtered Data")
    st.dataframe(dff, use_container_width=True, height=400)
    st.download_button("Download filtered CSV", data=dff.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_data.csv", mime="text/csv")

else:
    st.info("Upload a CSV file from the sidebar to get started.")
