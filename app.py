
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import resolve_col, coerce_purchase_series, safe_percent, to_datetime_series

st.set_page_config(page_title="Merged CSV Dashboard", layout="wide")

st.title("📊 Merged Customer Dashboard")
st.caption("Upload the merged CSV from Apps Script. Works with either system-style or human-friendly headers.")

with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown("---")
    st.write("Chart metric")
    y_metric_mode = st.radio("Y-axis", ["Purchases","Conversion Rate"], horizontal=True)

@st.cache_data(show_spinner=False)
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def agg_by_dim(d: pd.DataFrame, dim: str):
    g = d.groupby(dim, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    g["conv_rate"] = (g["purchases"]/g["rows"]).replace([np.inf,-np.inf], np.nan)*100
    g["conv_rate"] = g["conv_rate"].fillna(0.0)
    g[dim] = g[dim].astype(str).replace("nan","(blank)")
    return g

def leaderboard(d: pd.DataFrame, dim: str, sort_metric: str, min_rows: int, top_n: int):
    g = agg_by_dim(d, dim)
    g = g[g["rows"] >= min_rows]
    if g.empty:
        st.info(f"No groups meet min rows ≥ {min_rows}.")
        return
    g_sorted = g.sort_values(sort_metric, ascending=False).head(top_n)
    show = g_sorted.rename(columns={dim: dim, "rows":"Visitors", "purchases":"Purchases", "conv_rate":"Conversion %"})
    show["Conversion %"] = show["Conversion %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(show, use_container_width=True, hide_index=True)
    y = "purchases" if y_metric_mode == "Purchases" else "conv_rate"
    y_title = "Purchases" if y == "purchases" else "Conversion Rate (%)"
    fig = px.bar(g_sorted, x=dim, y=y, text=g_sorted[y].round(2), title=f"Top {top_n} by {'Purchases' if y=='purchases' else 'Conversion %'}")
    if y == "conv_rate":
        fig.update_yaxes(title=y_title)
    st.plotly_chart(fig, use_container_width=True)

def multi_dim_ranker(d: pd.DataFrame, dims: list[str], sort_key: str, min_rows: int, top_n: int, show_all: bool):
    if not dims:
        st.info("Choose 1–4 dimensions.")
        return
    dims = dims[:4]
    g = d.groupby(dims, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    g["conv_rate"] = (g["purchases"]/g["rows"]).replace([np.inf,-np.inf], np.nan)*100
    for dim in dims:
        g[dim] = g[dim].astype(str).replace("nan","(blank)")
    st.caption(f"Found **{len(g)}** combinations before filters.")
    g = g[g["rows"] >= min_rows]
    st.caption(f"**{len(g)}** combinations meet min rows ≥ {min_rows}.")
    if g.empty:
        st.info("No combinations after filtering.")
        return
    g_sorted = g.sort_values(sort_key, ascending=False)
    if not show_all:
        g_sorted = g_sorted.head(top_n)
    show = g_sorted.rename(columns={"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"})
    show["Conversion %"] = show["Conversion %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download ranked combos", data=g_sorted.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_combinations.csv", mime="text/csv")

if uploaded:
    df = load_df(uploaded)

    # Resolve columns
    email_col = resolve_col(df, "EMAIL")
    purchase_col = resolve_col(df, "PURCHASE")
    date_col = resolve_col(df, "DATE")
    order_count_col = resolve_col(df, "ORDER_COUNT")
    first_date_col = resolve_col(df, "FIRST_ORDER_DATE")
    last_date_col  = resolve_col(df, "LAST_ORDER_DATE")
    revenue_col    = resolve_col(df, "REVENUE")
    skus_col       = resolve_col(df, "SKUS")
    recent_sku_col = resolve_col(df, "MOST_RECENT_SKU")

    seg_keys = ["AGE_RANGE","CHILDREN","GENDER","HOMEOWNER","MARRIED","NET_WORTH","INCOME_RANGE","CREDIT_RATING"]
    seg_cols = []
    for k in seg_keys:
        c = resolve_col(df, k)
        if c: seg_cols.append(c)

    if purchase_col is None:
        st.error("Could not find a Purchase column. Expected one of: PURCHASE, Purchase, purchased, Buyer, is_buyer.")
        st.stop()

    df["_PURCHASE"] = coerce_purchase_series(df, purchase_col)

    # DATE for filters/trend
    if date_col and date_col in df.columns:
        df["_DATE"] = to_datetime_series(df[date_col])
    elif last_date_col and last_date_col in df.columns:
        df["_DATE"] = to_datetime_series(df[last_date_col])
    else:
        df["_DATE"] = pd.NaT

    with st.expander("🔎 Filters", expanded=True):
        dff = df.copy()
        # Date filter
        if not dff["_DATE"].dropna().empty:
            mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
            if pd.notna(mind) and pd.notna(maxd):
                start, end = st.date_input("Date range", (mind.date(), maxd.date()))
                include_undated = st.checkbox("Include rows with no date", value=True)
                if not isinstance(start, tuple):
                    mask = (dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end)))
                    if include_undated:
                        mask = mask | dff["_DATE"].isna()
                    dff = dff[mask]

        # SKU contains
        sku_search = st.text_input("SKU contains (optional)")
        if skus_col and sku_search:
            dff = dff[dff[skus_col].astype(str).str.contains(sku_search, case=False, na=False)]

        # Recent SKU filter
        if recent_sku_col:
            opts = sorted([x for x in dff[recent_sku_col].dropna().astype(str).unique() if x.strip()])
            sel = st.multiselect("Most Recent SKU (optional)", opts)
            if sel:
                dff = dff[dff[recent_sku_col].astype(str).isin(sel)]

        # Revenue slider
        if revenue_col:
            rev = pd.to_numeric(dff[revenue_col], errors="coerce").fillna(0)
            rmin, rmax = float(rev.min()), float(rev.max())
            lo, hi = st.slider("Revenue range (sum per person)", min_value=0.0, max_value=max(1.0, rmax), value=(0.0, max(1.0, rmax)))
            dff = dff[(rev >= lo) & (rev <= hi)]

        # Seg filters
        for c in seg_cols:
            opts = sorted([x for x in dff[c].dropna().unique().tolist() if str(x).strip()])
            sel = st.multiselect(f"Filter by {c}", opts)
            if sel:
                dff = dff[dff[c].isin(sel)]
        st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

    # KPIs
    total_rows = len(dff)
    total_p = int(dff["_PURCHASE"].sum())
    conv = safe_percent(total_p, total_rows)
    cols = st.columns(4)
    cols[0].metric("Total People", f"{total_rows:,}")
    cols[1].metric("Purchases", f"{total_p:,}")
    cols[2].metric("Conversion Rate", f"{conv:.2f}%")
    if revenue_col:
        tot_rev = pd.to_numeric(dff[revenue_col], errors="coerce").fillna(0).sum()
        cols[3].metric("Total Revenue", f"{tot_rev:,.2f}")

    st.markdown("---")

    # Leaderboards
    st.subheader("🏆 Leaderboards")
    if seg_cols:
        left, right = st.columns(2)
        with left:
            dim = st.selectbox("Dimension", options=seg_cols, index=0)
            sort_metric = st.selectbox("Sort by", options=["conv_rate","purchases","rows"], format_func=lambda x: {"conv_rate":"Conversion %","purchases":"Purchases","rows":"Visitors"}[x])
            min_rows = st.number_input("Min visitors per group", min_value=1, value=30, step=1)
            top_n = st.slider("Top N", 3, 100, 10, 1)
            leaderboard(dff, dim, sort_metric, min_rows, top_n)
        with right:
            st.markdown("**Multi-D Combinations**")
            dims = st.multiselect("Choose up to 4", options=seg_cols, default=[seg_cols[0]])
            metric2 = st.selectbox("Metric", options=["conv_rate","purchases","rows"], index=0, format_func=lambda x: {"conv_rate":"Conversion %","purchases":"Purchases","rows":"Visitors"}[x])
            min_rows2 = st.number_input("Min visitors per combo", min_value=1, value=20, step=1, key="mr2")
            top_n2 = st.slider("Top N combos", 3, 500, 50, 1, key="tn2")
            show_all = st.checkbox("Show ALL combos", value=False)
            multi_dim_ranker(dff, dims, metric2, min_rows2, top_n2, show_all)
    else:
        st.info("No segmentation columns detected. Include columns like Gender, Income Range, Net Worth, Homeowner, Credit Rating.")

    st.markdown("---")

    # Trend
    if not dff["_DATE"].isna().all():
        st.subheader("📈 Trend Over Time")
        ts = dff.copy()
        ts["date_only"] = ts["_DATE"].dt.date
        line = ts.groupby("date_only")["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
        line["conv_rate"] = (line["purchases"]/line["rows"])*100
        y = "purchases" if y_metric_mode == "Purchases" else "conv_rate"
        y_title = "Purchases" if y == "purchases" else "Conversion Rate (%)"
        fig = px.line(line, x="date_only", y=y, markers=True, title="Trend Over Time")
        fig.update_layout(yaxis_title=y_title, xaxis_title="Date", margin=dict(l=10,r=10,b=40,t=60))
        st.plotly_chart(fig, use_container_width=True)

    # Table + Export
    st.subheader("📥 Data (filtered)")
    st.dataframe(dff, use_container_width=True, height=420)
    st.download_button("Download filtered CSV", data=dff.to_csv(index=False).encode("utf-8"), file_name="filtered_data.csv", mime="text/csv")

else:
    st.info("Upload your merged CSV to begin.")
