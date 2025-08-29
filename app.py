
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import resolve_col, coerce_purchase, to_datetime_series, safe_percent, explode_skus, clean_sku_token

st.set_page_config(page_title="Ranked Customer Dashboard", layout="wide")

st.title("📊 Ranked Customer Dashboard")
st.caption("Upload the merged CSV. Use attribute dropdowns (All = group; pick value = filter). Ranks every combo by your chosen metric.")

with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown("---")
    metric_choice = st.radio("Sort metric", ["Conversion %","Purchases","Visitors"], horizontal=False)

@st.cache_data(show_spinner=False)
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

if uploaded:
    df = load_df(uploaded)

    # Resolve columns
    email_col = resolve_col(df, "EMAIL")
    purchase_col = resolve_col(df, "PURCHASE")
    if email_col is None or purchase_col is None:
        st.error("Missing EMAIL or PURCHASE column.")
        st.stop()

    df["_PURCHASE"] = coerce_purchase(df, purchase_col)

    date_col = resolve_col(df, "DATE") or resolve_col(df, "LAST_ORDER_DATE")
    if date_col and date_col in df.columns:
        df["_DATE"] = to_datetime_series(df[date_col])
    else:
        df["_DATE"] = pd.NaT

    skus_col = resolve_col(df, "SKUS")
    recent_sku_col = resolve_col(df, "MOST_RECENT_SKU")

    # Attribute candidates
    candidate_segs = {
        "Age Range": resolve_col(df, "AGE_RANGE"),
        "Income Range": resolve_col(df, "INCOME_RANGE"),
        "Net Worth": resolve_col(df, "NET_WORTH"),
        "Credit Rating": resolve_col(df, "CREDIT_RATING"),
        "Gender": resolve_col(df, "GENDER"),
        "Homeowner": resolve_col(df, "HOMEOWNER"),
        "Married": resolve_col(df, "MARRIED"),
        "Children": resolve_col(df, "CHILDREN"),
    }
    seg_cols_present = {lbl: col for lbl, col in candidate_segs.items() if col is not None}

    # Global Filters
    with st.expander("🔎 Global Filters", expanded=True):
        dff = df.copy()
        if not dff["_DATE"].dropna().empty:
            mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
            start, end = st.date_input("Date range", (mind.date(), maxd.date()))
            include_undated = st.checkbox("Include rows with no date", value=True)
            if not isinstance(start, tuple):
                mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
                if include_undated:
                    mask = mask | dff["_DATE"].isna()
                dff = dff[mask]

        sku_search = st.text_input("SKU contains (optional)")
        if skus_col and sku_search:
            dff = dff[dff[skus_col].astype(str).str.contains(sku_search, case=False, na=False)]

        if recent_sku_col:
            opts = sorted([x for x in dff[recent_sku_col].dropna().astype(str).unique() if x.strip()])
            sel = st.multiselect("Most Recent SKU (optional)", opts)
            if sel:
                dff = dff[dff[recent_sku_col].astype(str).isin(sel)]

        st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

    # Attributes (All=group; specific=filter)
    st.subheader("Attributes")
    attr_selections = {}
    group_candidates = []
    for label, col in seg_cols_present.items():
        values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
        choice = st.selectbox(label, options=["All"] + values, index=0, key=f"attr_{label}")
        if choice != "All":
            attr_selections[col] = [choice]   # filter
        else:
            group_candidates.append((label, col))

    # Apply selected filters
    for col, sel_vals in attr_selections.items():
        dff = dff[dff[col].isin(sel_vals)]

    # Determine group columns
    MAX_GROUP_ATTRS = 5
    if len(group_candidates) > MAX_GROUP_ATTRS:
        st.warning(f"Too many attributes on 'All' ({{len(group_candidates)}}). Pick up to {MAX_GROUP_ATTRS} to define rows.")
        labels = [lbl for lbl,_ in group_candidates]
        priority = ["Age Range","Income Range","Credit Rating","Net Worth","Homeowner","Gender","Married","Children"]
        ordered = [lbl for lbl in priority if lbl in labels] + [lbl for lbl in labels if lbl not in priority]
        default_select = ordered[:MAX_GROUP_ATTRS]
        chosen_labels = st.multiselect("Group by", options=labels, default=default_select, help="These columns define the row combinations.")
        group_cols = [dict(group_candidates)[lbl] for lbl in chosen_labels][:MAX_GROUP_ATTRS]
    else:
        group_cols = [col for _, col in group_candidates]

    if not group_cols:
        group_cols = ["__ALL__"]
        dff["__ALL__"] = "All"

    # Controls
    st.subheader("🏆 Ranked Conversion Table")
    c1, c2 = st.columns(2)
    with c1:
        min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)
    with c2:
        top_n = st.slider("Top N", 3, 2000, 100, 1)

    # Build groups
    g = dff.groupby(group_cols, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    g["conv_rate"] = (g["purchases"]/g["rows"]).replace([np.inf,-np.inf], np.nan)*100
    g = g[g["rows"] >= min_rows]

    # Top SKUs per group
    if skus_col and skus_col in dff.columns and not dff[dff["_PURCHASE"]==1].empty:
        skux = explode_skus(dff, skus_col)
        skux["__SKU"] = skux["__SKU"].apply(clean_sku_token)
        skux = skux.dropna(subset=["__SKU"])
        if group_cols != ["__ALL__"]:
            sku_counts = skux.groupby(group_cols + ["__SKU"]).size().reset_index(name="sku_buyers")
            # map from group key -> list of (sku, count)
            tmp = {}
            for _, r in sku_counts.iterrows():
                key = tuple(r[c] for c in group_cols)
                tmp.setdefault(key, []).append((r["__SKU"], int(r["sku_buyers"])))
            top_skus = []
            for _, r in g.iterrows():
                key = tuple(r[c] for c in group_cols)
                arr = sorted(tmp.get(key, []), key=lambda x: x[1], reverse=True)[:10]
                top_skus.append(", ".join([f"{sku} ({cnt})" for sku, cnt in arr]) if arr else "")
            g["Top SKUs (purchasers)"] = top_skus
        else:
            sku_counts = skux.groupby(["__SKU"]).size().reset_index(name="sku_buyers").sort_values("sku_buyers", ascending=False)
            g["Top SKUs (purchasers)"] = ", ".join([f"{r['__SKU']} ({int(r['sku_buyers'])})" for _, r in sku_counts.head(10).iterrows()])
    else:
        g["Top SKUs (purchasers)"] = ""

    # Sort and display
    sort_key = {"Conversion %":"conv_rate","Purchases":"purchases","Visitors":"rows"}[metric_choice]
    g_sorted = g.sort_values(sort_key, ascending=False).head(top_n)

    disp = g_sorted.rename(columns={"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"})
    disp["Conversion %"] = disp["Conversion %"].map(lambda x: f"{x:.2f}%")
    ordered_cols = group_cols + ["Visitors","Purchases","Conversion %","Top SKUs (purchasers)"]
    disp = disp[ordered_cols]
    st.dataframe(disp, use_container_width=True, hide_index=True)

else:
    st.info("Upload the merged CSV to begin.")
