import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from utils import resolve_col
from itertools import combinations

st.set_page_config(page_title="Ranked Customer Dashboard — DuckDB", layout="wide")
st.title("📊 Ranked Customer Dashboard (Fast • DuckDB)")
st.caption("Counts each person in every qualifying group (1..Max depth). Uses GROUPING SETS. SKU columns come from Most Recent SKU only.")

# ---------------- Sidebar ----------------
with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown('---')
    metric_choice = st.radio("Sort metric", ["Conversion %","Purchases","Visitors"], index=0)
    max_depth = st.slider('Max combo depth', 1, 4, 2, 1)
    top_n = st.slider('Top N', 10, 1000, 50, 10)

@st.cache_data(show_spinner=False)
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_datetime_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(s)))

if uploaded:
    # --------------- Load & resolve columns ---------------
    df = load_df(uploaded)
    email_col    = resolve_col(df, 'EMAIL')
    purchase_col = resolve_col(df, 'PURCHASE')
    date_col     = resolve_col(df, 'DATE')
    msku_col     = resolve_col(df, 'MOST_RECENT_SKU')
    state_col    = resolve_col(df, 'PERSONAL_STATE')  # <- State attribute
    if email_col is None or purchase_col is None:
        st.error('Missing EMAIL or PURCHASE column.')
        st.stop()

    # Purchase -> 0/1
    s = df[purchase_col]
    if pd.api.types.is_numeric_dtype(s):
        df['_PURCHASE'] = (s.fillna(0) > 0).astype(int)
    else:
        vals = s.astype(str).str.strip().str.lower()
        yes = {'1','true','t','yes','y','buyer','purchased'}
        df['_PURCHASE'] = vals.isin(yes).astype(int)

    # Optional date parsing for filters
    df['_DATE'] = to_datetime_series(df[date_col]) if date_col else pd.NaT

    # --------------- Attribute map (labels -> actual CSV columns) ---------------
    seg_map = {
        'Age':           resolve_col(df, 'AGE_RANGE'),
        'Income':        resolve_col(df, 'INCOME_RANGE'),
        'Net worth':     resolve_col(df, 'NET_WORTH'),
        'Credit rating': resolve_col(df, 'CREDIT_RATING'),
        'Gender':        resolve_col(df, 'GENDER'),
        'Homeowner':     resolve_col(df, 'HOMEOWNER'),
        'Married':       resolve_col(df, 'MARRIED'),
        'Children':      resolve_col(df, 'CHILDREN'),
        'State':         state_col,                     # <- included like the others
    }
    seg_map = {lbl: col for lbl, col in seg_map.items() if col is not None}
    seg_cols = list(seg_map.values())

    # --------------- Filters ---------------
    with st.expander('🔎 Filters', expanded=True):
        dff = df.copy()

        # Treat 'U' as missing for these attributes
        for lbl, col in seg_map.items():
            if lbl in ('Gender', 'Credit rating') and col in dff.columns:
                dff.loc[dff[col].astype(str).str.upper().str.strip() == 'U', col] = pd.NA

        # Date filter
        if not dff['_DATE'].dropna().empty:
            mind, maxd = pd.to_datetime(dff['_DATE'].dropna().min()), pd.to_datetime(dff['_DATE'].dropna().max())
            c1,c2 = st.columns(2)
            with c1:
                start, end = st.date_input('Date range', (mind.date(), maxd.date()))
            with c2:
                include_undated = st.checkbox('Include no-date', value=True)
            if not isinstance(start, tuple):
                mask = dff['_DATE'].between(pd.to_datetime(start), pd.to_datetime(end))
                if include_undated:
                    mask = mask | dff['_DATE'].isna()
                dff = dff[mask]

        # SKU substring filter
        sku_search = st.text_input('Most Recent SKU contains (optional)')
        if msku_col and sku_search:
            dff = dff[dff[msku_col].astype(str).str.contains(sku_search, case=False, na=False)]

        # Include/exclude + selections
        selections = {}
        include_flags = {}
        if seg_cols:
            st.markdown('**Attributes**')
            cols = st.columns(3)
            idx = 0
            for label, col in seg_map.items():
                with cols[idx % 3]:
                    mode = st.selectbox(f'{label}: mode', options=['Include', 'Do not include'], index=0, key=f'mode_{label}')
                    include_flags[col] = (mode == 'Include')
                    values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    sel = st.multiselect(label, options=values, default=[], help='Empty = All')
                    if sel:
                        selections[col] = sel
                idx += 1
            for col, vals in selections.items():
                dff = dff[dff[col].isin(vals)]

        st.caption(f'Rows after filters: **{len(dff):,}** / {len(df):,}')

    # --------------- Build GROUPING SETS ---------------
    include_cols = [c for c in seg_cols if include_flags.get(c, True)]
    required_cols = [col for col, vals in selections.items() if len(vals)>0 and include_flags.get(col, True)]

    con = duckdb.connect()
    con.register('t', dff)

    attrs = [c for c in include_cols if c in dff.columns]
    req_set = set(required_cols)

    sets = []
    for d in range(1, max_depth+1):
        for s in combinations(attrs, d):
            if req_set.issubset(set(s)):
                sets.append('(' + ','.join([f"\"{c}\"" for c in s]) + ')')

    if not sets:
        if required_cols:
            sets.append('(' + ','.join([f"\"{c}\"" for c in required_cols]) + ')')
        else:
            sets.append('(' + ','.join([f"\"{c}\"" for c in attrs[:1]]) + ')')

    grouping_sets_sql = ',\n'.join(sets)

    # Top SKUs (global among purchasers)
    top_skus = []
    if msku_col and msku_col in dff.columns:
        top_skus = con.execute(
            f'SELECT "{msku_col}" AS sku, COUNT(*) AS c '
            f'FROM t WHERE _PURCHASE=1 AND "{msku_col}" IS NOT NULL AND TRIM("{msku_col}")<>\'\' '
            f'GROUP BY 1 ORDER BY c DESC LIMIT 11'
        ).fetchdf()['sku'].astype(str).tolist()

    # Per-SKU buyer counts (no "SKU:" prefix)
    pieces = []
    for sku in top_skus:
        s_escaped = sku.replace("'", "''")
        pieces.append(
            'SUM(CASE WHEN "{msku}"=\'{sku}\' AND _PURCHASE=1 THEN 1 ELSE 0 END) AS "{label}"'
            .format(msku=msku_col, sku=s_escaped, label=sku)
        )
    sku_sums = ',\n  '.join(pieces)
    sku_sql = ("\n  ," + sku_sums) if sku_sums else ''

    depth_expr = ' + '.join([f'CASE WHEN "{c}" IS NULL THEN 0 ELSE 1 END' for c in attrs]) if attrs else '0'
    attrs_sql = ', '.join([f'"{c}"' for c in attrs]) if attrs else "'All' AS overall"

    sql = f"""
SELECT
  {attrs_sql},
  COUNT(*) AS Visitors,
  SUM(_PURCHASE) AS Purchases,
  100.0 * SUM(_PURCHASE) / NULLIF(COUNT(*),0) AS conv_rate,
  ({depth_expr}) AS Depth{sku_sql}
FROM t
GROUP BY GROUPING SETS (
  {grouping_sets_sql}
)
HAVING COUNT(*) >= ?
    """

    # --------------- Run & sort ---------------
    st.subheader('🏆 Ranked Conversion Table')
    c1, c2 = st.columns(2)
    with c1:
        min_rows = st.number_input('Minimum Visitors per group', min_value=1, value=30, step=1)
    with c2:
        pass

    res = con.execute(sql, [int(min_rows)]).fetchdf()
    sort_key = {'Conversion %':'conv_rate','Purchases':'Purchases','Visitors':'Visitors'}[metric_choice]
    res = res.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)

    # SKU columns in same order as top_skus
    sku_cols = [c for c in top_skus if c in res.columns]

    # --------------- Display (use CSV headers as-is) ---------------
    disp = res.copy()
    disp.insert(0, 'Rank', np.arange(1, len(disp) + 1))
    disp = disp.rename(columns={'conv_rate':'Conversion'})
    disp['Conversion'] = res['conv_rate'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else '')

    def _fmt_int(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ''
            return f"{int(round(float(v))):,}"
        except Exception:
            return v

    for c in ['Visitors','Purchases','Depth']:
        if c in disp.columns:
            disp[c] = disp[c].map(_fmt_int)
    for sc in sku_cols:
        disp[sc] = disp[sc].map(_fmt_int)

    # Attribute order (map labels -> actual CSV col names)
    logical_attr_order = ['Gender','Age','Homeowner','Married','Children','Credit rating','Income','Net worth','State']
    ordered_attr_cols = [seg_map[lbl] for lbl in logical_attr_order if lbl in seg_map and seg_map[lbl] in disp.columns]

    table_cols = ['Rank','Visitors','Purchases','Conversion'] + ordered_attr_cols + sku_cols

    # --------------- Render & Download ---------------
    st.dataframe(disp[table_cols], use_container_width=True, hide_index=True)

    csv_out = disp[table_cols]
    st.download_button('Download ranked combinations (CSV)',
                       data=csv_out.to_csv(index=False).encode('utf-8'),
                       file_name='ranked_combinations.csv', mime='text/csv')

else:
    st.info('Upload the merged CSV to begin.')
