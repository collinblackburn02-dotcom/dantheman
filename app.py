
import streamlit as st, pandas as pd, numpy as np, duckdb, plotly.express as px
from utils import resolve_col
from itertools import combinations

st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")

# Header
c1,c2=st.columns([0.12,0.88])
with c1: 
    try: st.image("logo.png",use_column_width=True)
    except: pass
with c2:
    st.markdown("<h1 style='margin-bottom:0'>Heavenly Health — Customer Insights</h1>", unsafe_allow_html=True)
    st.caption("Fast, ranked customer segments powered by DuckDB (GROUPING SETS).")

with st.sidebar:
    up = st.file_uploader("Upload merged CSV", type=["csv"])
    metric_choice = st.radio("Sort / Map metric", ["Conversion %","Purchases","Visitors","Revenue / Visitor"], index=0)
    max_depth = st.slider("Max combo depth",1,4,2,1)
    top_n = st.slider("Top N",10,1000,50,10)

@st.cache_data
def load_df(f):
    df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    return df

if up:
    df = load_df(up)
    email_col = resolve_col(df,"EMAIL")
    purchase_col = resolve_col(df,"PURCHASE")
    date_col = resolve_col(df,"DATE")
    sku_col = resolve_col(df,"MOST_RECENT_SKU")
    state_col = resolve_col(df,"PERSONAL_STATE")
    revenue_col = resolve_col(df,"REVENUE")

    # Purchase flag
    s = df[purchase_col].astype(str).str.lower().str.strip()
    df["_PURCHASE"] = s.isin({"1","true","t","yes","y","buyer","purchased"}).astype(int)
    df["_DATE"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0) if revenue_col else 0.0

    # Attributes map (include PERSONAL_STATE as "State")
    seg_map = {
        "Age": resolve_col(df,"AGE_RANGE"),
        "Income": resolve_col(df,"INCOME_RANGE"),
        "Net Worth": resolve_col(df,"NET_WORTH"),
        "Credit": resolve_col(df,"CREDIT_RATING"),
        "Gender": resolve_col(df,"GENDER"),
        "Homeowner": resolve_col(df,"HOMEOWNER"),
        "Married": resolve_col(df,"MARRIED"),
        "Children": resolve_col(df,"CHILDREN"),
        "State": state_col,
    }
    seg_map = {k:v for k,v in seg_map.items() if v is not None}
    seg_cols = list(seg_map.values())

    # ---- Filters ----
    with st.expander("🔎 Filters", expanded=True):
        dff = df.copy()
        # treat U as missing
        for k,col in seg_map.items():
            if k in ("Gender","Credit") and col in dff.columns:
                dff.loc[dff[col].astype(str).str.upper().str.strip()=="U", col] = pd.NA
        # date
        if not dff["_DATE"].dropna().empty:
            minD,maxD=dff["_DATE"].min(), dff["_DATE"].max()
            c1,c2=st.columns(2)
            with c1: dr = st.date_input("Date range", (minD.date(), maxD.date()))
            with c2: include_undated = st.checkbox("Include no-date rows", True)
            if isinstance(dr,tuple):
                mask = dff["_DATE"].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))
                if include_undated: mask = mask | dff["_DATE"].isna()
                dff = dff[mask]
        # sku search
        q = st.text_input("Most Recent SKU contains (optional)")
        if sku_col and q:
            dff = dff[dff[sku_col].astype(str).str.contains(q, case=False, na=False)]
        # include/do not include
        selections, include_flags = {}, {}
        if seg_cols:
            cols = st.columns(3); i=0
            for label,col in seg_map.items():
                with cols[i%3]:
                    mode = st.selectbox(f"{label}: mode",["Include","Do not include"], key=f"mode_{label}")
                    include_flags[col] = (mode=="Include")
                    vals = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    sel = st.multiselect(label, options=vals, default=[])
                    if sel: selections[col]=sel
                i+=1
            for col,vals in selections.items():
                dff = dff[dff[col].isin(vals)]
        st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

    include_cols = [c for c in seg_cols if include_flags.get(c,True)]
    required_cols = [c for c,v in selections.items() if len(v)>0 and include_flags.get(c,True)]

    # ---- DuckDB compute with GROUPING SETS ----
    con = duckdb.connect(); con.register("t", dff)
    attrs = [c for c in include_cols if c in dff.columns]

    sets=[]; req_set=set(required_cols)
    for d in range(1,max_depth+1):
        for s in combinations(attrs,d):
            if req_set.issubset(set(s)):
                sets.append("(" + ",".join([f'\"{c}\"' for c in s]) + ")")
    if not sets: sets=["()"]
    grouping_sets_sql = ",\n  ".join(sets)

    # fixed SKU order
    fixed_skus = ["ECO","FIR1","FIR2","RL2","TRA2","COM2","OUT2","RL500MID-W","RL900PRO-B","RL900PRO-W"]
    sku_sums = ""
    if sku_col and sku_col in dff.columns:
        parts=[f"SUM(CASE WHEN \"{sku_col}\"='{sku.replace(\"'\",\"''\")}' AND _PURCHASE=1 THEN 1 ELSE 0 END) AS \"SKU:{sku}\"" for sku in fixed_skus]
        sku_sums = ",\n      " + ",\n      ".join(parts)

    depth_expr = " + ".join([f'CASE WHEN \"{c}\" IS NULL THEN 0 ELSE 1 END' for c in attrs]) if attrs else "0"
    revenue_sql = "SUM(_REVENUE) AS revenue,\n      1.0 * SUM(_REVENUE) / NULLIF(COUNT(*),0) AS rpv" if \"_REVENUE\" in dff.columns else "0.0 AS revenue,\n      0.0 AS rpv"
    attrs_sql = ", ".join([f'\"{c}\"' for c in attrs]) if attrs else "'All' AS overall"

    sql = f"""
    SELECT
      {attrs_sql},
      COUNT(*) AS Visitors,
      SUM(_PURCHASE) AS Purchases,
      100.0 * SUM(_PURCHASE) / NULLIF(COUNT(*),0) AS conv_rate,
      ({depth_expr}) AS Depth,
      {revenue_sql}
      {sku_sums}
    FROM t
    GROUP BY GROUPING SETS (
      {grouping_sets_sql}
    )
    HAVING COUNT(*) >= ?
    """
    st.subheader("🏆 Ranked Conversion Table")
    min_rows = st.number_input("Minimum Visitors per group",1,100000,30,1)
    res = con.execute(sql,[int(min_rows)]).fetchdf()

    sort_key_map={\"Conversion %\":\"conv_rate\",\"Purchases\":\"Purchases\",\"Visitors\":\"Visitors\",\"Revenue / Visitor\":\"rpv\"}
    key = sort_key_map[metric_choice]
    res = res.sort_values(key, ascending=False).head(top_n).reset_index(drop=True)
    res.insert(0,\"Rank\",np.arange(1,len(res)+1))
    res[\"Conversion %\"]=res[\"conv_rate\"].map(lambda x: f\"{x:.2f}%\" if pd.notnull(x) else \"\")

    # clean blanks
    for c in attrs:
        if c in res.columns:
            res[c]=res[c].fillna(\"\").replace(\"None\",\"\")

    sku_cols=[c for c in [f\"SKU:{s}\" for s in fixed_skus] if c in res.columns]
    cols=[\"Rank\",\"Visitors\",\"Purchases\",\"Conversion %\",\"Depth\"] + sku_cols + [c for c in attrs]
    st.dataframe(res[cols], use_container_width=True, hide_index=True)

    # Map
    if state_col and state_col in dff.columns:
        st.subheader(\"🗺️ State Map\")
        agg = dff.groupby(state_col).agg(Visitors=(email_col,\"count\"), Purchases=(\"_PURCHASE\",\"sum\"), Revenue=(\"_REVENUE\",\"sum\")).reset_index()
        agg[\"conv_rate\"]=100.0*agg[\"Purchases\"]/agg[\"Visitors\"].replace(0,np.nan)
        agg[\"rpv\"]=agg[\"Revenue\"]/agg[\"Visitors\"].replace(0,np.nan)
        color = {\"Conversion %\":\"conv_rate\",\"Purchases\":\"Purchases\",\"Visitors\":\"Visitors\",\"Revenue / Visitor\":\"rpv\"}[metric_choice]
        fig = px.choropleth(agg, locations=state_col, locationmode=\"USA-states\", color=color, scope=\"usa\", color_continuous_scale=\"YlOrBr\")
        fig.update_layout(margin={\"l\":0,\"r\":0,\"t\":0,\"b\":0})
        st.plotly_chart(fig, use_container_width=True)
