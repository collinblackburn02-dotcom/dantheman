
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
from utils import resolve_col
from itertools import combinations

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")
try:
    st.logo("logo.png")
except Exception:
    pass

st.title("✨ Heavenly Health — Customer Insights")
st.caption("Ranked customer segments using DISTINCT visitors. Clean headers. SKUs on the right.")

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.markdown("### Data Source")
    gh_url = st.text_input(
        "GitHub CSV (raw) URL",
        placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/file.csv",
        help=(
            "Paste a RAW GitHub CSV URL. If provided, you don't need to upload a file."
        ),
    )
    st.markdown("**— or —**")
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])

    st.markdown("---")
    metric_choice = st.radio("Sort / Map metric", ["Conversion %", "Purchases", "Visitors", "Revenue / Visitor"], index=0)
    max_depth = st.slider("Max combo depth", 1, 4, 2, 1)
    top_n = st.slider("Top N", 10, 1000, 50, 10)
    min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)

    st.markdown("---")
    # Memory protection: cap attribute cardinality
    max_levels = st.slider(
        "Max unique values per attribute (cap)",
        min_value=2,
        max_value=200,
        value=50,
        step=1,
        help="Attributes with more distinct values than this are excluded to prevent blowups."
    )

    st.markdown("---")
    cache_ttl = st.number_input("Cache (minutes) for GitHub file", min_value=0, value=15, step=5)
    reload_clicked = st.button("🔄 Force Reload")

# -------------------------------------------------
# Data loading
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_df_from_github(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def _load_df_from_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_df(gh: str | None, up):
    if gh:
        try:
            if reload_clicked:
                _load_df_from_github.clear()
            if cache_ttl and cache_ttl > 0:
                minute_bucket = pd.Timestamp.utcnow().floor(f"{int(cache_ttl)}min")
                return _load_df_from_github(gh + f"?t={minute_bucket.isoformat()}")
            return _load_df_from_github(gh)
        except Exception as e:
            st.error("Failed to load CSV from GitHub URL.\n\n" + str(e))
            st.stop()
    if up is not None:
        return _load_df_from_upload(up)
    return None

df = load_df(gh_url.strip() if gh_url else None, uploaded)
if df is None:
    st.info("Paste a GitHub RAW CSV URL or upload a file to begin.")
    st.stop()

# -------------------------------------------------
# Resolve Columns
# -------------------------------------------------
email_col = resolve_col(df, "EMAIL")
purchase_col = resolve_col(df, "PURCHASE")
date_col = resolve_col(df, "DATE")
msku_col = resolve_col(df, "MOST_RECENT_SKU")
state_col = resolve_col(df, "PERSONAL_STATE")
revenue_col = resolve_col(df, "REVENUE")

if email_col is None or purchase_col is None:
    st.error("Missing EMAIL or PURCHASE column.")
    st.stop()

# Purchase as 0/1
s = df[purchase_col]
if pd.api.types.is_numeric_dtype(s):
    df["_PURCHASE"] = (s.fillna(0) > 0).astype(int)
else:
    vals = s.astype(str).str.strip().str.lower()
    yes = {"1", "true", "t", "yes", "y", "buyer", "purchased"}
    df["_PURCHASE"] = vals.isin(yes).astype(int)

# Date & Revenue
if date_col:
    try:
        df["_DATE"] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        df["_DATE"] = pd.NaT
else:
    df["_DATE"] = pd.NaT

if revenue_col:
    df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)
else:
    df["_REVENUE"] = 0.0

# Friendly attribute map
seg_map = {
    "Age": resolve_col(df, "AGE_RANGE"),
    "Income": resolve_col(df, "INCOME_RANGE"),
    "Net worth": resolve_col(df, "NET_WORTH"),
    "Credit rating": resolve_col(df, "CREDIT_RATING"),
    "Gender": resolve_col(df, "GENDER"),
    "Homeowner": resolve_col(df, "HOMEOWNER"),
    "Married": resolve_col(df, "MARRIED"),
    "Children": resolve_col(df, "CHILDREN"),
    "State": state_col,
}
seg_map = {disp: col for disp, col in seg_map.items() if col is not None}
seg_cols = list(seg_map.values())
friendly_label = {v: k for k, v in seg_map.items()}

# -------------------------------------------------
# Filters
# -------------------------------------------------
with st.expander("🔎 Filters", expanded=True):
    dff = df.copy()

    # Treat 'U' as missing for Gender and Credit
    for disp, col in seg_map.items():
        if disp in ("Gender", "Credit rating") and col in dff.columns:
            dff.loc[dff[col].astype(str).str.upper().str.strip() == "U", col] = pd.NA

    # Date filter
    if not dff["_DATE"].dropna().empty:
        mind = pd.to_datetime(dff["_DATE"].dropna().min())
        maxd = pd.to_datetime(dff["_DATE"].dropna().max())
        c1, c2 = st.columns(2)
        with c1:
            start, end = st.date_input("Date range", (mind.date(), maxd.date()))
        with c2:
            include_undated = st.checkbox("Include no-date rows", value=True)
        if not isinstance(start, tuple):
            mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
            if include_undated:
                mask = mask | dff["_DATE"].isna()
            dff = dff[mask]

    # SKU contains
    sku_search = st.text_input("Most Recent SKU contains (optional)")
    if msku_col and sku_search:
        dff = dff[dff[msku_col].astype(str).str.contains(sku_search, case=False, na=False)]

    # Attribute level cap to avoid blowups
    too_tall = []
    for disp, col in seg_map.items():
        nlv = dff[col].nunique(dropna=True)
        if nlv > max_levels:
            too_tall.append((disp, col, nlv))
    if too_tall:
        st.warning("Skipped high-cardinality attributes: " + ", ".join([f"{d} ({n} levels)" for d, c, n in too_tall]))
        seg_cols_capped = [c for d, c in seg_map.items() if d not in [tt[0] for tt in too_tall]]
    else:
        seg_cols_capped = seg_cols

    # Attribute filters with Include/Do not include
    selections = {}
    include_flags = {}
    if seg_cols_capped:
        st.markdown("**Attributes**")
        cols = st.columns(3)
        idx = 0
        for display, col in seg_map.items():
            if col not in seg_cols_capped:
                continue
            with cols[idx % 3]:
                mode = st.selectbox(f"{display}: mode", options=["Include", "Do not include"], index=0, key=f"mode_{display}")
                include_flags[col] = (mode == "Include")
                values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                sel = st.multiselect(display, options=values, default=[], help="Empty = All")
                if sel:
                    selections[col] = sel
            idx += 1
        for col, vals in selections.items():
            dff = dff[dff[col].isin(vals)]

    st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# -------------------------------------------------
# Deduplicate to one row per visitor (latest by date)
# -------------------------------------------------
# Sorting once then dropping duplicates is cheaper than DISTINCT repeatedly.
if date_col:
    dff_sorted = dff.sort_values(by="_DATE", ascending=False, kind="mergesort")
else:
    dff_sorted = dff.copy()
dff_one = dff_sorted.drop_duplicates(subset=[email_col], keep="first").reset_index(drop=True)

# -------------------------------------------------
# DuckDB: register and compute
# -------------------------------------------------
con = duckdb.connect()
con.register("t", dff_one)

# Build attribute list honoring include flags
include_cols = [c for c in seg_cols_capped if include_flags.get(c, True)]
attrs = [c for c in include_cols if c in dff_one.columns]

# Build GROUPING SETS that include any required columns (from explicit selections)
required_cols = [col for col, vals in selections.items() if len(vals) > 0 and include_flags.get(col, True)]
req_set = set(required_cols)
sets = []
for d in range(1, max_depth + 1):
    for combo in combinations(attrs, d):
        if req_set.issubset(set(combo)):
            sets.append("(" + ",".join([f'"{c}"' for c in combo]) + ")")
if not sets:
    if required_cols:
        sets.append("(" + ",".join([f'"{c}"' for c in required_cols]) + ")")
    else:
        if attrs:
            sets.append("(" + ",".join([f'"{c}"' for c in attrs[:1]]) + ")")
        else:
            sets.append("()")
grouping_sets_sql = ",\n".join(sets)

# Top SKUs by distinct buyers overall (on deduped table)
sku_sums = ""
sku_cols_order = []
if msku_col and msku_col in dff_one.columns:
    top_skus = con.execute(f"""
        SELECT "{msku_col}" AS sku,
               COUNT(CASE WHEN _PURCHASE=1 THEN 1 END) AS buyers
        FROM t
        WHERE "{msku_col}" IS NOT NULL AND TRIM("{msku_col}")<>''
        GROUP BY 1
        ORDER BY buyers DESC
        LIMIT 11
    """).fetchdf()["sku"].astype(str).tolist()
    if top_skus:
        sku_cols_order = top_skus
        parts = []
        for sku in top_skus:
            s_escaped = sku.replace("'", "''")
            parts.append(
                f'SUM(CASE WHEN "{msku_col}"='{s_escaped}' AND _PURCHASE=1 THEN 1 ELSE 0 END) AS "{sku}"'
            )
        sku_sums = ",\n  " + ",\n  ".join(parts)

# Depth expression
if attrs:
    depth_expr = " + ".join([f'CASE WHEN "{c}" IS NULL THEN 0 ELSE 1 END' for c in attrs])
else:
    depth_expr = "0"

# Build SELECT safely
select_parts = []
if attrs:
    select_parts.extend([f'"{c}"' for c in attrs])
select_parts.append('COUNT(*) AS Visitors')
select_parts.append('SUM(_PURCHASE) AS Purchases')
select_parts.append('100.0 * SUM(_PURCHASE) / NULLIF(COUNT(*),0) AS conv_rate')
select_parts.append(f'({depth_expr}) AS Depth')
if revenue_col:
    select_parts.append('SUM(_REVENUE) AS revenue')
    select_parts.append('1.0 * SUM(_REVENUE) / NULLIF(COUNT(*),0) AS rpv')
else:
    select_parts.append('0.0 AS revenue')
    select_parts.append('0.0 AS rpv')
if sku_sums:
    select_parts.append(sku_sums.lstrip(",\n "))
select_clause = ",\n  ".join(select_parts)

sql = f"""
SELECT
  {select_clause}
FROM t
{'GROUP BY GROUPING SETS (' + grouping_sets_sql + ')' if attrs else ''}
HAVING COUNT(*) >= ?
"""

# Execute with robust error handling
try:
    res = con.execute(sql, [int(min_rows)]).fetchdf()
except Exception as e:
    st.error(
        "Query failed (possibly due to memory). Try lowering Max combo depth, Top N, or the attribute cap.\n\n"
        + str(e)
    )
    st.stop()

# -------------------------------------------------
# Display prep: clean headers and numbers
# -------------------------------------------------
sort_key_map = {"Conversion %": "conv_rate", "Purchases": "Purchases", "Visitors": "Visitors", "Revenue / Visitor": "rpv"}
res = res.sort_values(sort_key_map[metric_choice], ascending=False).head(top_n).reset_index(drop=True)

# SKU columns
sku_cols = [c for c in sku_cols_order if c in res.columns]

# Friendly column renames for attributes and metrics
rename_map = {c: friendly_label.get(c, c) for c in attrs}
rename_map.update({"conv_rate": "Conversion %", "rpv": "Revenue / Visitor", "revenue": "Revenue"})

disp = res.copy().rename(columns=rename_map)
disp.insert(0, "Rank", np.arange(1, len(disp) + 1))

def fmt_int(v):
    if pd.isna(v):
        return ""
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return str(v)

for col in ["Visitors", "Purchases", "Depth"]:
    if col in disp.columns:
        disp[col] = disp[col].map(fmt_int)

if "Conversion %" in disp.columns:
    disp["Conversion %"] = res["conv_rate"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
if "Revenue / Visitor" in disp.columns:
    disp["Revenue / Visitor"] = res["rpv"].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

# Clean attribute values
for c in rename_map.values():
    if c in disp.columns:
        disp[c] = disp[c].fillna("").replace("None", "")

# SKU formatting (blank for 0)
for sc in sku_cols:
    disp[sc] = res[sc].map(lambda x: "" if pd.isna(x) or int(round(float(x))) == 0 else fmt_int(x))

# Column order: core, attributes, then SKUs at far right
attribute_display_cols = [rename_map[c] for c in attrs]
table_cols = ["Rank", "Visitors", "Purchases", "Conversion %", "Depth"] + attribute_display_cols + sku_cols

def highlight_conv(s):
    return ["font-weight: bold" if s.name == "Conversion %" else "" for _ in s]

st.dataframe(disp[table_cols].style.apply(highlight_conv, axis=0), use_container_width=True, hide_index=True)

# Download CSV
csv_out = res.copy().rename(columns=rename_map)
csv_out.insert(0, "Rank", np.arange(1, len(csv_out) + 1))
csv_cols = ["Rank", "Visitors", "Purchases", "conv_rate", "Depth", "rpv", "revenue"] + attrs + sku_cols
csv_out = csv_out[csv_cols].rename(columns={"conv_rate": "Conversion % (0-100)", "rpv": "Revenue / Visitor", "revenue": "Revenue", **{c: rename_map[c] for c in attrs}})
st.download_button("Download ranked combinations (CSV)", data=csv_out.to_csv(index=False).encode("utf-8"), file_name="ranked_combinations.csv", mime="text/csv")

# -------------------------------------------------
# Map by State (distinct visitors using deduped table)
# -------------------------------------------------
if state_col and state_col in dff_one.columns:
    st.subheader("🗺️ State Map")
    metric_for_map = sort_key_map[metric_choice]
    # dff_one is already 1 row per email
    map_df = dff_one.copy()
    map_df[state_col] = map_df[state_col].astype(str).str.upper().str.strip()
    agg = map_df.groupby(state_col).agg(
        Visitors=(email_col, "count"),
        Purchases=("_PURCHASE", "sum"),
        Revenue=("_REVENUE", "sum"),
    ).reset_index()
    agg["conv_rate"] = 100.0 * agg["Purchases"] / agg["Visitors"].replace(0, np.nan)
    agg["rpv"] = agg["Revenue"] / agg["Visitors"].replace(0, np.nan)
    fig = px.choropleth(
        agg,
        locations=state_col,
        locationmode="USA-states",
        color=metric_for_map,
        scope="usa",
        color_continuous_scale="YlOrBr",
        labels={"conv_rate": "Conversion %", "rpv": "Revenue / Visitor"},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
