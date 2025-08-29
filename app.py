
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import coerce_purchase_series, pick_default_purchase_col, safe_percent, detect_date_col, coerce_datetime

st.set_page_config(page_title="CSV Dashboard", layout="wide")

st.markdown("## 📊 Customer Analytics Dashboard")
st.caption("Upload a CSV, choose your purchase column, and explore KPIs, filters, and charts. Export filtered data anytime.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload your CSV to begin.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    st.divider()
    st.write("**Optional:** configure columns if auto-detection isn't perfect.")

@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Trim whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    return df

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
        # Purchase column selection
        default_purchase_col = pick_default_purchase_col(df) or st.selectbox("Purchase Indicator Column (choose)", list(df.columns))
        purchase_col = st.selectbox("Purchase Indicator Column", 
                                    options=[default_purchase_col] + [c for c in df.columns if c != default_purchase_col]) if default_purchase_col else st.selectbox("Purchase Indicator Column", list(df.columns))
        st.caption("If numeric: values > 0 = purchased. If text: values like 'yes/true/1/buyer' = purchased.")

        # Date column (optional)
        guessed_date = detect_date_col(df)
        date_col = st.selectbox("Date Column (optional)", ["" ] + list(df.columns), index=(list(df.columns).index(guessed_date)+1 if guessed_date in df.columns else 0))

        # Common dimension filters toggles
        st.markdown("**Enable filters for these columns (if present):**")
        filter_cols = []
        for c in ["GENDER","AGE_RANGE","INCOME_RANGE","NET_WORTH","HOMEOWNER","SKIPTRACE_CREDIT_RATING","MARRIED","CHILDREN"]:
            if c in df.columns and st.checkbox(c, value=(c in ["GENDER","INCOME_RANGE","SKIPTRACE_CREDIT_RATING","HOMEOWNER"])):
                filter_cols.append(c)

        st.divider()
        st.write("**Chart settings**")
        y_metric_mode = st.radio("Y-axis metric", ["Purchases", "Conversion Rate"], horizontal=True)

    # Coerce purchase series
    try:
        df["_PURCHASE"] = coerce_purchase_series(df, purchase_col)
    except Exception as e:
        st.error(f"Could not interpret purchase column `{purchase_col}`. Error: {e}")
        st.stop()

    # Coerce date if provided
    if date_col and date_col in df.columns:
        df["_DATE"] = coerce_datetime(df, date_col)
    else:
        df["_DATE"] = pd.NaT

    # Apply filters
    with st.expander("🔎 Filters", expanded=True):
        dff = df.copy()
        for c in filter_cols:
            dff = get_filter(dff, c, f"Filter by {c}")
        # Optional date range filter
        if not dff["_DATE"].isna().all():
            min_d = pd.to_datetime(dff["_DATE"].min())
            max_d = pd.to_datetime(dff["_DATE"].max())
            if pd.notna(min_d) and pd.notna(max_d):
                start, end = st.date_input("Date range", (min_d.date(), max_d.date()))
                if isinstance(start, tuple):  # streamlit quirk if no default
                    pass
                else:
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

    # Helper: plot either purchases or conversion rate by a dimension
    def plot_by_dim(d: pd.DataFrame, dim: str, title: str):
        if dim not in d.columns:
            return
        grp = d.groupby(dim, dropna=False).agg(rows=(" _PURCHASE".strip(), "count"), purchases=("_PURCHASE","sum")).reset_index()
        grp["conv_rate"] = grp.apply(lambda r: safe_percent(r["purchases"], r["rows"]), axis=1)
        y = "purchases" if y_metric_mode == "Purchases" else "conv_rate"
        y_title = "Purchases" if y == "purchases" else "Conversion Rate (%)"
        fig = px.bar(grp.sort_values(y, ascending=False), x=dim, y=y, title=title, text=grp[y].round(2))
        fig.update_layout(yaxis_title=y_title, xaxis_title=dim, margin=dict(l=10,r=10,b=40,t=60))
        st.plotly_chart(fig, use_container_width=True)

    # Layout: two columns of key breakdowns if present
    c1, c2 = st.columns(2)

    with c1:
        if "INCOME_RANGE" in dff.columns:
            plot_by_dim(dff, "INCOME_RANGE", "Performance by Income Range")
        if "SKIPTRACE_CREDIT_RATING" in dff.columns:
            plot_by_dim(dff, "SKIPTRACE_CREDIT_RATING", "Performance by Credit Rating")

    with c2:
        if "NET_WORTH" in dff.columns:
            plot_by_dim(dff, "NET_WORTH", "Performance by Net Worth")
        if "GENDER" in dff.columns:
            plot_by_dim(dff, "GENDER", "Performance by Gender")

    st.divider()

    # Time series (if date available)
    if not dff["_DATE"].isna().all():
        ts = dff.copy()
        ts["date_only"] = ts["_DATE"].dt.date
        line = ts.groupby("date_only").agg(rows=(" _PURCHASE".strip(),"count"), purchases=("_PURCHASE","sum")).reset_index()
        line["conv_rate"] = line.apply(lambda r: safe_percent(r["purchases"], r["rows"]), axis=1)
        y = "purchases" if y_metric_mode == "Purchases" else "conv_rate"
        y_title = "Purchases" if y == "purchases" else "Conversion Rate (%)"
        fig = px.line(line, x="date_only", y=y, markers=True, title="Trend Over Time")
        fig.update_layout(yaxis_title=y_title, xaxis_title="Date", margin=dict(l=10,r=10,b=40,t=60))
        st.plotly_chart(fig, use_container_width=True)

    # Show table + download
    st.subheader("📥 Filtered Data")
    st.dataframe(dff, use_container_width=True, height=400)
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")

else:
    st.info("Upload a CSV file from the sidebar to get started.")
