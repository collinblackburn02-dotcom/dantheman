
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import resolve_col, coerce_purchase, to_datetime_series, safe_percent, explode_skus

st.set_page_config(page_title="Ranked Customer Dashboard", layout="wide")

st.title("📊 Ranked Customer Dashboard")
st.caption("Upload the merged CSV. Choose attributes to rank by; compare any combinations; click Purchases to see people and a reactive ZIP map.")

with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown("---")
    y_metric_mode = st.radio("Metric", ["Conversion %","Purchases","Visitors"], horizontal=False)
    st.markdown("---")
    st.write("Optional: upload ZIP centroids CSV (zip,lat,lon) for the map")
    zip_lookup = st.file_uploader("ZIP centroid CSV", type=["csv"], key="zip")

@st.cache_data(show_spinner=False)
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_zip_lookup(file):
    try:
        z = pd.read_csv(file, dtype={"zip": str})
        z["zip"] = z["zip"].str.zfill(5)
        return z[["zip","lat","lon"]]
    except Exception:
        return None

if uploaded:
    df = load_df(uploaded)

    # Resolve columns (system or human-friendly)
    email_col = resolve_col(df, "EMAIL")
    if email_col is None:
        st.error("Missing email column. Expected one of EMAIL/Email/email.")
        st.stop()

    purchase_col = resolve_col(df, "PURCHASE")
    if purchase_col is None:
        st.error("Missing Purchase column. Expected PURCHASE/Purchase/etc.")
        st.stop()

    df["_PURCHASE"] = coerce_purchase(df, purchase_col)

    date_col = resolve_col(df, "DATE") or resolve_col(df, "LAST_ORDER_DATE")
    if date_col and date_col in df.columns:
        df["_DATE"] = to_datetime_series(df[date_col])
    else:
        df["_DATE"] = pd.NaT

    revenue_col = resolve_col(df, "REVENUE")
    skus_col = resolve_col(df, "SKUS")
    recent_sku_col = resolve_col(df, "MOST_RECENT_SKU")
    zip_col = resolve_col(df, "ZIP")

    # Available attributes to choose from
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
    seg_cols_present = {label:col for label,col in candidate_segs.items() if col is not None}

    with st.expander("🔎 Global Filters", expanded=True):
        dff = df.copy()
        # Date filter (keep undated by default)
        if not dff["_DATE"].dropna().empty:
            mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
            start, end = st.date_input("Date range", (mind.date(), maxd.date()))
            include_undated = st.checkbox("Include rows with no date", value=True)
            if not isinstance(start, tuple):
                mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
                if include_undated:
                    mask = mask | dff["_DATE"].isna()
                dff = dff[mask]

        # SKU contains
        sku_search = st.text_input("SKU contains (optional)")
        if skus_col and sku_search:
            dff = dff[dff[skus_col].astype(str).str.contains(sku_search, case=False, na=False)]

        # Most recent SKU filter
        if recent_sku_col:
            opts = sorted([x for x in dff[recent_sku_col].dropna().astype(str).unique() if x.strip()])
            sel = st.multiselect("Most Recent SKU (optional)", opts)
            if sel:
                dff = dff[dff[recent_sku_col].astype(str).isin(sel)]

        # Revenue range
        if revenue_col:
            rev = pd.to_numeric(dff[revenue_col], errors="coerce").fillna(0)
            lo, hi = float(rev.min()), float(rev.max())
            rsel = st.slider("Revenue range (sum per person)", 0.0, max(1.0, hi), (0.0, max(1.0, hi)))
            dff = dff[(rev >= rsel[0]) & (rev <= rsel[1])]

        st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

    # Attribute selection UI
    st.subheader("Attributes to include in ranking")
    chosen_attrs = []
    attr_selections = {}
    cols = st.columns(3)
    idx = 0
    for label, col in seg_cols_present.items():
        with cols[idx % 3]:
            use_attr = st.checkbox(f"Include {label}", value=False, key=f"use_{label}")
            if use_attr:
                chosen_attrs.append((label, col))
                # value multiselect (no selection = all)
                values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                sel_vals = st.multiselect(f"{label} values", values, default=[])
                attr_selections[col] = sel_vals
        idx += 1

    # Apply attribute value selections (no selection means include all)
    for col, sel_vals in attr_selections.items():
        if sel_vals:
            dff = dff[dff[col].isin(sel_vals)]

    # Build ranking
    st.subheader("🏆 Ranked Conversion Table")
    left, right = st.columns([1,1])
    with left:
        min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)
    with right:
        top_n = st.slider("Top N", 3, 1000, 100, 1)

    # Grouping
    group_cols = [col for _, col in chosen_attrs]
    if not group_cols:
        group_cols = ["__ALL__"]
        dff["__ALL__"] = "All"

    grp = dff.groupby(group_cols, dropna=False)["_PURCHASE"].agg(rows="count", purchases="sum").reset_index()
    grp["conv_rate"] = (grp["purchases"]/grp["rows"]).replace([np.inf,-np.inf], np.nan)*100
    grp = grp[grp["rows"] >= min_rows]

    # Build SKU counts per group (from purchasers). If no SKU column, leave blank.
    if skus_col and skus_col in dff.columns:
        skux = explode_skus(dff, skus_col)
        if group_cols != ["__ALL__"]:
            sku_counts = skux.groupby(group_cols + ["__SKU"]).size().reset_index(name="sku_buyers")
            top_sku_strings = []
            # Create a dict key for fast join
            def key_from_row(row):
                return tuple(row[g] for g in group_cols)
            # For each group, collect top SKUs
            grp_keys = grp[group_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
            sku_map = {}
            for _, r in sku_counts.iterrows():
                k = tuple(r[g] for g in group_cols)
                sku_map.setdefault(k, []).append((r["__SKU"], int(r["sku_buyers"])))
            for k in grp_keys:
                arr = sorted(sku_map.get(k, []), key=lambda x: x[1], reverse=True)[:10]
                s = ", ".join([f"{sku} ({cnt})" for sku, cnt in arr]) if arr else ""
                top_sku_strings.append(s)
            grp["Top SKUs (purchasers)"] = top_sku_strings
        else:
            sku_counts = skux.groupby(["__SKU"]).size().reset_index(name="sku_buyers").sort_values("sku_buyers", ascending=False)
            s = ", ".join([f"{r['__SKU']} ({int(r['sku_buyers'])})" for _, r in sku_counts.head(10).iterrows()])
            grp["Top SKUs (purchasers)"] = s
    else:
        grp["Top SKUs (purchasers)"] = ""

    # Sort by chosen metric
    sort_key = {"Conversion %":"conv_rate","Purchases":"purchases","Visitors":"rows"}[y_metric_mode]
    grp_sorted = grp.sort_values(sort_key, ascending=False).head(top_n)


    # Build a clean DataFrame for display
    disp_df = grp_sorted.copy()
    # Pretty column titles
    rename_cols = {"rows":"Visitors","purchases":"Purchases","conv_rate":"Conversion %"} 
    disp_df = disp_df.rename(columns=rename_cols)
    # Truncate top SKUs for readability
    def _ellipsize(s, maxlen=120):
        try:
            s = str(s)
            return s if len(s) <= maxlen else s[:maxlen-1] + '…'
        except Exception:
            return s
    disp_df["Top SKUs (purchasers)"] = disp_df["Top SKUs (purchasers)"].apply(lambda x: _ellipsize(x, 90))
    disp_df["Conversion %"] = disp_df["Conversion %"].map(lambda x: f"{x:.2f}%")
    # Reorder columns
    ordered_cols = [*([c for c in group_cols]), "Visitors", "Purchases", "Conversion %", "Top SKUs (purchasers)"]
    disp_df = disp_df[ordered_cols]
    st.subheader("📋 Ranked Table")
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    # Simple selector to focus a row (clean alternative to per-row buttons)
    combo_labels = []
    for _, r in disp_df.iterrows():
        parts = [f"{c}={r[c]}" for c in group_cols] if group_cols != ["__ALL__"] else ["All"]
        combo_labels.append(f"{' ; '.join(parts)} • Purchases: {int(r['Purchases'])} • Visitors: {int(r['Visitors'])}")
    if combo_labels:
        sel_label = st.selectbox("Focus row (optional)", options=["(none)"] + combo_labels, index=0)
        if sel_label != "(none)":
            idx = combo_labels.index(sel_label)
            # Save focus combo
            row = disp_df.iloc[idx]
            combo_vals = [str(row[c]) for c in group_cols]
            st.session_state["focus_combo"] = (tuple(zip(group_cols, combo_vals)), group_cols)
    # (Old custom row renderer removed in favor of clean table + selector)

    # Reactive purchaser list
    st.markdown("---")
    st.subheader("👥 Purchasers in selection")
    focus = st.session_state.get("focus_combo")
    if focus is not None:
        pairs, group_cols_current = focus
        # Build mask
        mask = dff["_PURCHASE"] == 1
        for col, val in pairs:
            if col != "__ALL__":
                mask = mask & (dff[col].astype(str) == str(val))
        buyers = dff[mask].copy()
        st.caption(f"{len(buyers):,} purchasers match: " + "; ".join([f"{col}={val}" for col,val in pairs if col!='__ALL__']))

        # Columns to show
        cols_to_show = [email_col]
        if resolve_col(df, "ORDER_COUNT"): cols_to_show.append(resolve_col(df, "ORDER_COUNT"))
        if resolve_col(df, "LAST_ORDER_DATE"): cols_to_show.append(resolve_col(df, "LAST_ORDER_DATE"))
        if revenue_col: cols_to_show.append(revenue_col)
        if recent_sku_col: cols_to_show.append(recent_sku_col)
        if skus_col: cols_to_show.append(skus_col)
        # plus active attributes
        cols_to_show += [c for c in group_cols_current if c != "__ALL__"]
        cols_to_show = list(dict.fromkeys(cols_to_show))  # dedupe preserve order

        # Pagination
        page_size = st.selectbox("Rows per page", [25,50,100,200], index=0)
        page = st.number_input("Page", min_value=1, value=1, step=1)
        start = (page-1)*page_size
        end = start + page_size
        st.dataframe(buyers[cols_to_show].iloc[start:end], use_container_width=True, height=360)
        st.download_button("Download purchaser list (CSV)", data=buyers[cols_to_show].to_csv(index=False).encode("utf-8"),
                           file_name="purchasers.csv", mime="text/csv")
        if st.button("Clear selection"):
            st.session_state["focus_combo"] = None
    else:
        st.info("Click a row's **View purchasers** to see the people list.")

    # ZIP heat/bubble map (reactive)
    st.markdown("---")
    st.subheader("🗺️ Purchaser Map by ZIP")
    if zip_col is None:
        st.info("No ZIP column detected. Add PERSONAL_ZIP / Billing Zip / Shipping Zip to your merged CSV, or map won't render.")
    else:
        # Build current purchaser set (either focused or all filtered purchasers)
        if focus is not None:
            pairs, group_cols_current = focus
            mask = dff["_PURCHASE"] == 1
            for col, val in pairs:
                if col != "__ALL__":
                    mask = mask & (dff[col].astype(str) == str(val))
            buyers = dff[mask].copy()
        else:
            buyers = dff[dff["_PURCHASE"] == 1].copy()

        buyers["__zip5"] = buyers[zip_col].astype(str).str.extract(r"(\d{5})", expand=False)
        zip_counts = buyers.groupby("__zip5").size().reset_index(name="purchasers")
        zip_counts = zip_counts[zip_counts["__zip5"].notna()]

        # Get zip centroids
        zlookup = load_zip_lookup(zip_lookup) if zip_lookup else None
        if zlookup is None:
            # Fallback: show top ZIP table and a bar chart instead of map
            st.warning("ZIP centroid file not provided. Upload a CSV with columns: zip,lat,lon to enable the map.")
            topz = zip_counts.sort_values("purchasers", ascending=False).head(50)
            st.dataframe(topz, use_container_width=True, height=320)
            fig = px.bar(topz, x="__zip5", y="purchasers", title="Top ZIPs by purchasers")
            st.plotly_chart(fig, use_container_width=True)
        else:
            z = zlookup.copy()
            merged = pd.merge(zip_counts, z, left_on="__zip5", right_on="zip", how="left")
            merged = merged.dropna(subset=["lat","lon"])
            if merged.empty:
                st.info("No ZIPs matched the centroid file.")
            else:
                fig = px.scatter_mapbox(
                    merged, lat="lat", lon="lon", size="purchasers", color="purchasers",
                    hover_name="__zip5", hover_data={"purchasers":True, "lat":False, "lon":False},
                    zoom=3, height=500
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload the merged CSV to begin.")
