
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from itertools import combinations
from utils import resolve_col

st.set_page_config(page_title="Heavenly Health — Customer Insights (v7.3.3 checkpoint)", layout="wide")

# Header
c1, c2 = st.columns([0.12, 0.88])
with c1:
    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        pass
with c2:
    st.markdown("<h1 style='margin-bottom:0'>Heavenly Health — Customer Insights</h1>", unsafe_allow_html=True)
    st.caption("v7.3.3 checkpoint — ranked segments, attribute filters, and SKU summaries.")

with st.sidebar:
    up = st.file_uploader("Upload merged CSV", type=["csv"])
    metric_choice = st.radio("Sort metric", ["Conversion %","Purchases","Visitors"], index=0)
    max_depth = st.slider("Max combo depth", 1, 4, 2, 1)
    top_n = st.slider("Top N rows", 10, 1000, 50, 10)

@st.cache_data(show_spinner=False)
def load_df(f):
    df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_dt(s: pd.Series):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(s)))

if up:
    df = load_df(up)
    email_col = resolve_col(df, "EMAIL")
    purchase_col = resolve_col(df, "PURCHASE")
    date_col = resolve_col(df, "DATE")
    msku_col = resolve_col(df, "MOST_RECENT_SKU")

    # Purchase flag
    s = df[purchase_col]
    if pd.api.types.is_numeric_dtype(s):
        df["_PURCHASE"] = (s.fillna(0) > 0).astype(int)
    else:
        vals = s.astype(str).str.strip().str.lower()
        yes = {"1","true","t","yes","y","buyer","purchased"}
        df["_PURCHASE"] = vals.isin(yes).astype(int)

    df["_DATE"] = to_dt(df[date_col]) if date_col else pd.NaT

    # Attribute map (NO map, no state here)
    seg_map = {
        "Age": resolve_col(df, "AGE_RANGE"),
        "Income": resolve_col(df, "INCOME_RANGE"),
        "Net Worth": resolve_col(df, "NET_WORTH"),
        "Credit": resolve_col(df, "CREDIT_RATING"),
        "Gender": resolve_col(df, "GENDER"),
        "Homeowner": resolve_col(df, "HOMEOWNER"),
        "Married": resolve_col(df, "MARRIED"),
        "Children": resolve_col(df, "CHILDREN"),
    }
    seg_map = {k:v for k,v in seg_map.items() if v is not None}
    seg_cols = [v for v in seg_map.values()]

    # -------- Filters --------
    with st.expander("🔎 Filters", expanded=True):
        dff = df.copy()

        # Treat 'U' as missing for Gender and Credit
        for k, col in seg_map.items():
            if k in ("Gender", "Credit") and col in dff.columns:
                dff.loc[dff[col].astype(str).str.upper().str.strip() == "U", col] = pd.NA

        # Date
        if not dff["_DATE"].dropna().empty:
            mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
            cA, cB = st.columns(2)
            with cA:
                start, end = st.date_input("Date range", (mind.date(), maxd.date()))
            with cB:
                include_undated = st.checkbox("Include no-date rows", True)
            if not isinstance(start, tuple):
                mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
                if include_undated:
                    mask = mask | dff["_DATE"].isna()
                dff = dff[mask]

        # SKU search (optional)
        sku_search = st.text_input("Most Recent SKU contains (optional)")
        if msku_col and sku_search:
            dff = dff[dff[msku_col].astype(str).str.contains(sku_search, case=False, na=False)]

        # Include / Do not include + value filters (3-col grid)
        selections = {}
        include_flags = {}
        if seg_cols:
            st.markdown("**Attributes**")
            cols = st.columns(3)
            idx = 0
            for label, col in seg_map.items():
                with cols[idx % 3]:
                    mode = st.selectbox(f"{label}: mode", options=["Include", "Do not include"], index=0, key=f"mode_{label}")
                    include_flags[col] = (mode == "Include")
                    values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    sel = st.multiselect(label, options=values, default=[], help="Empty = All")
                    if sel:
                        selections[col] = sel
                idx += 1
            for col, vals in selections.items():
                dff = dff[dff[col].isin(vals)]

        st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

    include_cols = [c for c in seg_cols if include_flags.get(c, True)]
    required_cols = [col for col, vals in selections.items() if len(vals)>0 and include_flags.get(col, True)]

    # -------- Compute rankings (pandas) --------
    # Build all grouping combinations up to max_depth that include required columns
    attrs = [c for c in include_cols if c in dff.columns]
    combos = []
    req_set = set(required_cols)
    for d in range(1, max_depth+1):
        for s in combinations(attrs, d):
            if req_set.issubset(set(s)):
                combos.append(list(s))
    if not combos:
        combos = [attrs[:1]] if attrs else [[]]

    rows = []
    for group in combos:
        if group:
            g = dff.groupby(group, dropna=False, as_index=False).agg(
                Visitors=(email_col, "count"),
                Purchases=("_PURCHASE", "sum"),
            )
            g["Depth"] = len(group)
            # Top SKUs text per group (among purchasers)
            if msku_col:
                # compute top 3 SKUs for each group
                sku_counts = dff.loc[dff["_PURCHASE"].eq(1)].groupby(group + [msku_col], dropna=False)[email_col].count().reset_index(name="c")
                top3 = (sku_counts.sort_values([*group, "c"], ascending=[True]*len(group)+[False])
                                 .groupby(group)
                                 .apply(lambda x: ", ".join([f"{str(r[msku_col])} ({int(r['c'])})" for _, r in x.head(5).iterrows()]))
                                 .reset_index(name="Top SKUs (purchasers)"))
                g = g.merge(top3, on=group, how="left")
            else:
                g["Top SKUs (purchasers)"] = ""
            rows.append(pd.DataFrame(g))
        else:
            # overall
            g = pd.DataFrame({
                "Visitors":[len(dff)],
                "Purchases":[int(dff["_PURCHASE"].sum())],
                "Depth":[0],
                "Top SKUs (purchasers)":[""]
            })
            rows.append(g)

    res = pd.concat(rows, ignore_index=True)
    res["conv_rate"] = 100.0 * res["Purchases"] / res["Visitors"].replace(0, np.nan)
    res = res.replace({None:""}).fillna("")

    sort_key = {"Conversion %":"conv_rate","Purchases":"Purchases","Visitors":"Visitors"}[metric_choice]
    res = res.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)

    # Rank & display columns
    res.insert(0, "Rank", np.arange(1, len(res)+1))
    res["Conversion %"] = res["conv_rate"].map(lambda x: f"{x:.2f}%" if x != "" else "")
    # Clean attribute blanks
    for col in attrs:
        if col in res.columns:
            res[col] = res[col].replace("None","").replace({np.nan:""})

    # Order: Rank, Visitors, Purchases, Conversion %, Top SKUs, then attributes
    attr_cols = [c for c in attrs]
    show = ["Rank","Visitors","Purchases","Conversion %","Top SKUs (purchasers)"] + attr_cols

    st.subheader("🏆 Ranked Conversion Table")
    st.dataframe(res[show], hide_index=True, use_container_width=True)

else:
    st.info("Upload the merged CSV to begin.")
