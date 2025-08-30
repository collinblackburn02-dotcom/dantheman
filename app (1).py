
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
from utils import resolve_col
from itertools import combinations

# -------------------------
# App Config & Header
# -------------------------
st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")
try:
    st.logo("logo.png")
except Exception:
    pass  # older Streamlit versions may not have st.logo

st.title("✨ Heavenly Health — Customer Insights")
st.caption("Fast, ranked customer segments powered by DuckDB (GROUPING SETS).")

# -------------------------
# Sidebar: Data source & controls
# -------------------------
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.markdown("### Data Source")
    gh_url = st.text_input(
        "GitHub CSV (raw) URL",
        placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/file.csv",
        help=(
            "Paste the *raw* URL of your CSV file from GitHub.\n"
            "Example: https://raw.githubusercontent.com/username/repo/main/data/merged.csv\n"
            "If provided, the app will auto-fetch from GitHub and you won't need to upload a file."
        ),
    )
    st.markdown("**— or —**")
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])

    st.markdown("---")
    metric_choice = st.radio(
        "Sort / Map metric",
        ["Conversion %", "Purchases", "Visitors", "Revenue / Visitor"],
        index=0,
    )
    max_depth = st.slider("Max combo depth", 1, 4, 2, 1)
    top_n = st.slider("Top N", 10, 1000, 50, 10)

    st.markdown("---")
    cache_ttl = st.number_input(
        "Cache (minutes) for GitHub file",
        min_value=0, value=15, step=5,
        help="Set to 0 to disable caching of the downloaded CSV."
    )
    reload_clicked = st.button("🔄 Force Reload")

# -------------------------
# Data loading
# -------------------------
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

def load_df(gh_url: str | None, uploaded_file):
    if gh_url:
        try:
            if reload_clicked:
                _load_df_from_github.clear()
            if cache_ttl and cache_ttl > 0:
                minute_bucket = pd.Timestamp.utcnow().floor(f"{int(cache_ttl)}min")
                return _load_df_from_github(gh_url + f"?t={minute_bucket.isoformat()}")
            return _load_df_from_github(gh_url)
        except Exception as e:
            st.error(f"Failed to load CSV from GitHub URL.\n\n{e}")
            st.stop()
    elif uploaded_file:
        return _load_df_from_upload(uploaded_file)
    else:
        return None

def to_datetime_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(s)))

df = load_df(gh_url.strip() if gh_url else None, uploaded)

if df is None:
    st.info("Paste a GitHub *raw* CSV URL in the sidebar **or** upload the merged CSV to begin.")
    st.stop()

# -------------------------
# Resolve key columns
# -------------------------
email_col = resolve_col(df, "EMAIL")
purchase_col = resolve_col(df, "PURCHASE")
date_col = resolve_col(df, "DATE")
msku_col = resolve_col(df, "MOST_RECENT_SKU")
state_col = resolve_col(df, "PERSONAL_STATE")
revenue_col = resolve_col(df, "REVENUE")

if email_col is None or purchase_col is None:
    st.error("Missing EMAIL or PURCHASE column.")
    st.stop()

# Coerce purchase to 0/1
s = df[purchase_col]
if pd.api.types.is_numeric_dtype(s):
    df["_PURCHASE"] = (s.fillna(0) > 0).astype(int)
else:
    vals = s.astype(str).str.strip().str.lower()
    yes = {"1", "true", "t", "yes", "y", "buyer", "purchased"}
    df["_PURCHASE"] = vals.isin(yes).astype(int)

df["_DATE"] = to_datetime_series(df[date_col]) if date_col else pd.NaT

# Numeric revenue; non-purchasers = 0 if revenue col exists
if revenue_col:
    df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)
else:
    df["_REVENUE"] = 0.0

# Attribute columns (State label shown as 'State')
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
# Drop missing
seg_map = {display: col for display, col in seg_map.items() if col is not None}
seg_cols = list(seg_map.values())
# For display: map real column name -> friendly name
friendly_label = {v: k for k, v in seg_map.items()}

# -------------------------
# Filters
# -------------------------
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

    # Attribute filters with Include/Do not include
    selections = {}
    include_flags = {}
    if seg_cols:
        st.markdown("**Attributes**")
        cols = st.columns(3)
        idx = 0
        for display, col in seg_map.items():
            with cols[idx % 3]:
                mode = st.selectbox(
                    f"{display}: mode",
                    options=["Include", "Do not include"],
                    index=0,
                    key=f"mode_{display}",
                )
                include_flags[col] = (mode == "Include")
                values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                sel = st.multiselect(display, options=values, default=[], help="Empty = All")
                if sel:
                    selections[col] = sel
            idx += 1
        for col, vals in selections.items():
            dff = dff[dff[col].isin(vals)]

    st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# Included attributes and required ones
include_cols = [c for c in seg_cols if include_flags.get(c, True)]
required_cols = [col for col, vals in selections.items() if len(vals) > 0 and include_flags.get(col, True)]

# -------------------------
# DuckDB compute (distinct visitors by email)
# -------------------------
con = duckdb.connect()
con.register("t", dff)

attrs = [c for c in include_cols if c in dff.columns]

req_set = set(required_cols)
sets = []
for d in range(1, max_depth + 1):
    for s_ in combinations(attrs, d):
        if req_set.issubset(set(s_)):
            sets.append("(" + ",".join([f'"{c}"' for c in s_]) + ")")
if not sets:
    if required_cols:
        sets.append("(" + ",".join([f'"{c}"' for c in required_cols]) + ")")
    else:
        if attrs:
            sets.append("(" + ",".join([f'"{c}"' for c in attrs[:1]]) + ")")
        else:
            sets.append("()")

grouping_sets_sql = ",\n".join(sets)

# Build SKU per-group distinct purchase counts (by email)
sku_sums = ""
top_skus = []
if msku_col and msku_col in dff.columns:
    top_skus = con.execute(f"""
        SELECT "{msku_col}" AS sku, COUNT(DISTINCT CASE WHEN _PURCHASE=1 THEN "{email_col}" END) AS buyers
        FROM t
        WHERE "{msku_col}" IS NOT NULL AND TRIM("{msku_col}")<>''
        GROUP BY 1
        ORDER BY buyers DESC
        LIMIT 11
    """).fetchdf()["sku"].astype(str).tolist()
    if top_skus:
        pieces = []
        for sku in top_skus:
            s_escaped = sku.replace("'", "''")
            pieces.append(
                f'COUNT(DISTINCT CASE WHEN "{msku_col}"=\'{s_escaped}\' AND _PURCHASE=1 THEN "{email_col}" END) AS "{sku}"'
            )
        sku_sums = ",\n  " + ",\n  ".join(pieces)

depth_expr = " + ".join([f'CASE WHEN "{c}" IS NULL THEN 0 ELSE 1 END' for c in attrs]) if attrs else "0"

# Revenue/RPV aggregates per distinct visitor
revenue_sql = (
    f'SUM(_REVENUE) AS revenue,\n  1.0 * SUM(_REVENUE) / NULLIF(COUNT(DISTINCT "{email_col}"),0) AS rpv'
    if revenue_col
    else "0.0 AS revenue, 0.0 AS rpv"
)

sql = f"""
SELECT
  {", ".join([f'"{c}"' for c in attrs])},
  COUNT(DISTINCT "{email_col}") AS Visitors,
  COUNT(DISTINCT CASE WHEN _PURCHASE=1 THEN "{email_col}" END) AS Purchases,
  100.0 * COUNT(DISTINCT CASE WHEN _PURCHASE=1 THEN "{email_col}" END)
        / NULLIF(COUNT(DISTINCT "{email_col}"),0) AS conv_rate,
  ({depth_expr}) AS Depth,
  {revenue_sql}
  {sku_sums}
FROM t
{'GROUP BY GROUPING SETS (' + grouping_sets_sql + ')' if attrs else ''}
HAVING COUNT(DISTINCT "{email_col}") >= ?
"""

# -------------------------
# Ranked table
# -------------------------
st.subheader("🏆 Ranked Conversion Table")
min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)
res = con.execute(sql, [int(min_rows)]).fetchdf()

# Sort & top-N
sort_key_map = {"Conversion %": "conv_rate", "Purchases": "Purchases", "Visitors": "Visitors", "Revenue / Visitor": "rpv"}
sort_key = sort_key_map[metric_choice]
res = res.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)

# -------- Display prep: clean headers & formatting --------
# Identify SKU columns (exactly the labels we aliased)
sku_cols = [c for c in res.columns if c not in set(["Visitors","Purchases","conv_rate","Depth","rpv","revenue"] + attrs)]

# Friendly labels for attribute columns
rename_map = {c: friendly_label.get(c, c) for c in attrs}
rename_map.update({"conv_rate": "Conversion %", "rpv": "Revenue / Visitor", "revenue": "Revenue"})

disp = res.copy().rename(columns=rename_map)

# Insert Rank
disp.insert(0, "Rank", np.arange(1, len(disp) + 1))

# Format numbers
def fmt_int(v):
    if pd.isna(v):
        return ""
    try:
        v = int(round(float(v)))
    except Exception:
        return str(v)
    return f"{v:,}"

# Core numeric columns
for col in ["Visitors", "Purchases", "Depth"]:
    if col in disp.columns:
        disp[col] = disp[col].map(fmt_int)

# Conversion %, RPV
if "Conversion %" in disp.columns:
    disp["Conversion %"] = res["conv_rate"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
if "Revenue / Visitor" in disp.columns:
    disp["Revenue / Visitor"] = res["rpv"].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

# Clean attribute values
for c in rename_map.values():
    if c in disp.columns:
        disp[c] = disp[c].fillna("").replace("None", "")

# SKU columns formatting to integers (blank if 0)
for sc in sku_cols:
    disp[sc] = res[sc].map(lambda x: "" if pd.isna(x) or int(round(float(x))) == 0 else fmt_int(x))

# Column order: Rank, Visitors, Purchases, Conversion %, Depth, ATTRIBUTES..., SKUs...
attribute_display_cols = [rename_map[c] for c in attrs]
table_cols = ["Rank", "Visitors", "Purchases", "Conversion %", "Depth"] + attribute_display_cols + sku_cols

# Style
def highlight_conv(s):
    return ["font-weight: bold" if s.name == "Conversion %" else "" for _ in s]

styled = disp[table_cols].style.apply(highlight_conv, axis=0)
st.dataframe(styled, use_container_width=True, hide_index=True)

# Download CSV (with clean headers)
csv_out = res.copy().rename(columns=rename_map)
csv_out.insert(0, "Rank", np.arange(1, len(csv_out) + 1))
csv_cols = ["Rank", "Visitors", "Purchases", "conv_rate", "Depth", "rpv", "revenue"] + attrs + sku_cols
csv_out = csv_out[csv_cols].rename(columns={"conv_rate": "Conversion % (0-100)", "rpv": "Revenue / Visitor", "revenue": "Revenue", **{c: rename_map[c] for c in attrs}})
st.download_button(
    "Download ranked combinations (CSV)",
    data=csv_out.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations.csv",
    mime="text/csv",
)

# -------------------------
# Map by State (distinct visitors)
# -------------------------
if state_col and state_col in dff.columns:
    st.subheader("🗺️ State Map")
    metric_for_map = sort_key_map[metric_choice]
    map_df = dff.copy()
    map_df[state_col] = map_df[state_col].astype(str).str.upper().str.strip()
    agg = map_df.groupby(state_col).agg(
        Visitors=(email_col, "nunique"),
        Purchases=("_PURCHASE", "sum"),
        Revenue=("_REVENUE", "sum"),
    ).reset_index()
    # Distinct conversion rate
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
