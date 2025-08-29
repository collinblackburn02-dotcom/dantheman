
import streamlit as st
import pandas as pd
import numpy as np
import itertools
from utils import resolve_col, coerce_purchase, to_datetime_series, explode_skus, clean_sku_token

st.set_page_config(page_title="Ranked Customer Dashboard", layout="wide")

st.title("📊 Ranked Customer Dashboard")
st.caption("Ranks **all combinations** (1..Max depth) within your filtered dataset. A purchaser counts in every qualifying group.")

with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown("---")
    metric_choice = st.radio("Sort metric", ["Conversion %","Purchases","Visitors"], horizontal=False)
    max_depth = st.slider("Max combo depth", 1, 4, 2, 1)

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
    date_col = resolve_col(df, "DATE")
    skus_col = None  # disabled list-based SKU parsing
    recent_sku_col = resolve_col(df, "MOST_RECENT_SKU")

    if email_col is None or purchase_col is None:
        st.error("Missing EMAIL or PURCHASE column.")
        st.stop()

    df["_PURCHASE"] = coerce_purchase(df, purchase_col)
    df["_DATE"] = to_datetime_series(df[date_col]) if date_col else pd.NaT

    # Attribute columns
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
    seg_cols = list(seg_map.values())

    # ---------- Filters (compact) ----------
    with st.expander("🔎 Filters", expanded=True):
        dff = df.copy()
        # Date range
        if not dff["_DATE"].dropna().empty:
            mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
            c1,c2,c3 = st.columns([1,1,1])
            with c1:
                start, end = st.date_input("Date range", (mind.date(), maxd.date()))
            with c2:
                include_undated = st.checkbox("Include no-date", value=True)
            if not isinstance(start, tuple):
                mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
                if include_undated:
                    mask = mask | dff["_DATE"].isna()
                dff = dff[mask]

        # SKU filters
        c1,c2 = st.columns([1,1])
        with c1:
            sku_search = st.text_input("SKU contains (optional)")
        with c2:
            if recent_sku_col:
                opts = sorted([x for x in dff[recent_sku_col].dropna().astype(str).unique() if x.strip()])
                sel = st.multiselect("Most Recent SKU", opts)
                if sel:
                    dff = dff[dff[recent_sku_col].astype(str).isin(sel)]
        if recent_sku_col and sku_search:
            dff = dff[dff[recent_sku_col].astype(str).str.contains(sku_search, case=False, na=False)]

        # Attribute multiselect filters in a compact 3-col grid
        if seg_cols:
            st.markdown("**Attributes**")
            cols = st.columns(3)
            selections = {}
            idx = 0
            for label, col in seg_map.items():
                with cols[idx % 3]:
                    values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    sel = st.multiselect(label, options=values, default=[], help="Empty = All")
                    if sel:
                        selections[col] = sel
                idx += 1
            # Apply
            for col, vals in selections.items():
                dff = dff[dff[col].isin(vals)]
            # Store selections for later display fill-in
            selected_singletons = {col: vals[0] for col, vals in selections.items() if len(vals)==1}

        st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

    # ---------- Enumerate all combos up to max_depth ----------
    st.subheader("🏆 Ranked Conversion Table")
    c1,c2,c3 = st.columns(3)
    with c1:
        min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)
    with c2:
        top_n = st.slider("Top N", 3, 10000, 500, 1)
    with c3:
        sku_top_k = st.slider("SKU columns (Top K)", 0, 25, 10, 1, help="Adds K most-purchased SKUs as columns")

    if not seg_cols:
        # Only overall row
        g = dff.groupby(["__ALL__"], dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
        g["conv_rate"] = (g["purchases"]/g["rows"])*100
        g["Depth"] = 0
        g["__ALL__"] = "All"
        results = g
        all_cols = ["__ALL__"]
    else:
        results_list = []
        # Prepare SKU exploded (from purchasers) once
        if recent_sku_col and recent_sku_col in dff.columns and not dff[dff["_PURCHASE"]==1].empty:
            skux = dff[dff["_PURCHASE"]==1][[recent_sku_col] + seg_cols].copy()
            skux = skux.rename(columns={recent_sku_col: "__SKU"})
            skux["__SKU"] = skux["__SKU"].apply(clean_sku_token)
            skux = skux.dropna(subset=["__SKU"])
        else:
            skux = None

        for k in range(1, max_depth+1):
            for subset in itertools.combinations(seg_cols, k):
                subset = list(subset)
                # Enforce: all selected attributes must be in the subset
                try:
                    req = set(required_cols)
                except Exception:
                    req = set()
                if not req.issubset(set(subset)):
                    continue
                g = dff.groupby(subset, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
                g["conv_rate"] = (g["purchases"]/g["rows"]).replace([np.inf,-np.inf], np.nan)*100
                g["Depth"] = k
                # Ensure all seg columns exist; fill with selected singleton value if available
                for col in seg_cols:
                    if col not in subset:
                        # Fill with singleton selection if present; else blank
                        try:
                            val = selected_singletons.get(col, "")
                        except Exception:
                            val = ""
                        g[col] = val
                # Top SKUs per group for this subset
                if skux is not None:
                    sc = skux.groupby(subset + ["__SKU"]).size().reset_index(name="sku_buyers")
                    # map
                    tmp = {}
                    for _, r in sc.iterrows():
                        key = tuple(r[c] for c in subset)
                        tmp.setdefault(key, []).append((r["__SKU"], int(r["sku_buyers"])))
                    top_skus = []
                    for _, r in g.iterrows():
                        key = tuple(r[c] for c in subset)
                        arr = sorted(tmp.get(key, []), key=lambda x: x[1], reverse=True)[:10]
                        top_skus.append(", ".join([f"{sku} ({cnt})" for sku, cnt in arr]) if arr else "")
                    g["Top SKUs (purchasers)"] = top_skus
                else:
                    g["Top SKUs (purchasers)"] = ""
                results_list.append(g)

        results = pd.concat(results_list, ignore_index=True)
        all_cols = seg_cols


    # Pivot SKUs into columns (Top K across filtered data)
    if recent_sku_col and 'Top SKUs (purchasers)' in results.columns and sku_top_k > 0:
        # Determine top K SKUs overall
        if 'skux' in locals() and skux is not None and not skux.empty:
            top_overall = skux['__SKU'].value_counts().head(sku_top_k).index.tolist()
            # For each subset depth, counts per group are embedded in results? Recompute fast per needed keys:
            # Build a key string for each row to merge counts
            def key_tuple(row, cols):
                return tuple(row[c] for c in cols if c in results.columns)
            # Create empty columns
            for sku in top_overall:
                results[f'SKU:{sku}'] = 0
            # For each depth, compute counts and fill
            for k in range(1, max_depth+1):
                # find rows at this depth by checking 'Depth'
                rows_k = results['Depth'] == k
                if not rows_k.any():
                    continue
                subset_cols = [c for c in seg_cols if (results.columns.isin([c]).any())]  # all seg cols exist in results
                # Identify which cols are non-empty indicators per row; but easier: for k depth, grouped earlier by specific subset.
                # We'll recompute counts per subset using skux for the specific subset columns that are non-empty in each row.
                # To avoid N^2, iterate unique group keys for those rows.
                # Determine actual subset cols by selecting seg_cols where that row has non-empty value (string not empty)
                sub_df = results[rows_k].copy()
                # For each unique pattern of which columns are set, group fill
                for _, r in sub_df.iterrows():
                    subset = [c for c in seg_cols if r.get(c, "") != ""]
                    if not subset:
                        continue
                    filt = skux.copy()
                    good = True
                    for c in subset:
                        filt = filt[filt[c].astype(str) == str(r[c])]
                        if filt.empty:
                            good = False
                            break
                    if not good or filt.empty:
                        continue
                    counts = filt['__SKU'].value_counts()
                    for sku in top_overall:
                        val = int(counts.get(sku, 0))
                        results.loc[(results['Depth']==k) & (results[subset].astype(str).eq([str(r[c]) for c in subset]).all(axis=1)), f'SKU:{sku}'] = val
        # Drop textual Top SKUs column in favor of pivot columns
        if 'Top SKUs (purchasers)' in results.columns:
            results = results.drop(columns=['Top SKUs (purchasers)'])
    # Apply min visitors and sort
    results = results[results["rows"] >= min_rows]
    sort_key = {"Conversion %":"conv_rate","Purchases":"purchases","Visitors":"rows"}[metric_choice]
    results = results.sort_values(sort_key, ascending=False).head(top_n)

    # Prepare display
    disp = results.rename(columns={"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"})
    disp["Conversion %"] = disp["Conversion %"].map(lambda x: f"{x:.2f}%")
    # Reorder columns: all attribute cols (using their original names), then Visitors, Purchases, Conversion, Depth, Top SKUs
    # Build ordered columns: attributes, then metrics, then SKU columns (if any)
    sku_cols = [c for c in disp.columns if c.startswith('SKU:')]
    base_cols = all_cols + ["Visitors","Purchases","Conversion %","Depth"]
    ordered_cols = base_cols + sku_cols
    disp = disp[ordered_cols]
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button("Download ranked combinations (CSV)", data=disp.to_csv(index=False).encode("utf-8"), file_name="ranked_combinations.csv", mime="text/csv")

else:
    st.info("Upload the merged CSV to begin.")
