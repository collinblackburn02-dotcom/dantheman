import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Ranked Customer Dashboard — Precomputed CSV", layout="wide")
st.title("📊 Ranked Customer Dashboard (Precomputed CSV)")
st.caption("Auto-loads your CSV from the repo. No upload needed. Skips first 3 rows, then filters/ranks/displays.")

# ================= Sidebar =================
with st.sidebar:
    st.markdown("### Controls")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    top_n = st.slider("Top N", 10, 2000, 50, 10)
    # Minimum visitors: floor 100, default 100
    min_rows = st.number_input("Minimum Visitors per group", min_value=100, value=100, step=1)

    st.markdown("---")
    with st.expander("Advanced (CSV location)"):
        st.write("If you rename or move the CSV, set the new repo-relative path here.")
        default_csv_path = "Copy of DAN_HHS - Sample.csv"
        override_path = st.text_input("CSV path (repo-relative)", value=default_csv_path, help="Example: data/yourfile.csv")
        skiprows = st.number_input("Header rows to skip", min_value=0, value=3, step=1)

# ================= Helpers =================
def resolve(df: pd.DataFrame, *candidates):
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = str(cand).strip().lower()
        if k in norm:
            return norm[k]
    return None

def find_attributes(df: pd.DataFrame):
    attr_map = {
        "Gender":               resolve(df, "Gender", "GENDER"),
        "Age":                  resolve(df, "Age_Range", "AGE_RANGE", "Age", "AGE"),
        "Homeowner":            resolve(df, "Homeowner", "HOMEOWNER"),
        "Married":              resolve(df, "Married", "MARRIED"),
        "Children":             resolve(df, "Children", "CHILDREN"),
        "Credit rating":        resolve(df, "Credit_Rating", "CREDIT_RATING", "Credit", "CREDIT"),
        "Income":               resolve(df, "Income_Range", "INCOME_RANGE", "Income", "INCOME"),
        "Net worth":            resolve(df, "New_Worth", "NET_WORTH", "NETWORTH", "Net_Worth"),
        "State":                resolve(df, "State", "PERSONAL_STATE", "STATE"),
        # Extra attributes
        "Ethnicity (skiptrace)": resolve(df, "SKIPTRACE_ETHNIC_CODE", "Skiptrace_Ethnic_Code"),
        "Department":            resolve(df, "DEPARTMENT", "Department"),
        "Seniority level":       resolve(df, "SENIORITY_LEVEL", "Seniority_Level", "Seniority"),
        "Skiptrace credit":      resolve(df, "SKIPTRACE_CREDIT_RATING", "Skiptrace_Credit_Rating"),
    }
    return {k: v for k, v in attr_map.items() if v is not None}

def fmt_int_series(s: pd.Series) -> pd.Series:
    return s.apply(lambda v: "" if pd.isna(v) else f"{int(round(float(v))):,}")

def looks_like_percent_strings(s: pd.Series) -> bool:
    if s.dtype == "O":
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        return (sample.str.contains("%").mean() > 0.6)
    return False

def load_repo_csv(path_str: str, skiprows: int) -> pd.DataFrame:
    base = Path.cwd()
    candidates = [
        base / path_str,
        base / "data" / Path(path_str).name,
        base / "datasets" / Path(path_str).name,
        base / "assets" / Path(path_str).name,
    ]
    last_err = None
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p, skiprows=skiprows)
                return df
        except Exception as e:
            last_err = e
    msg = f"Could not find/read CSV at any of these paths:\n" + "\n".join([str(c) for c in candidates])
    if last_err:
        msg += f"\n\nLast error: {last_err}"
    st.error(msg)
    st.stop()

# ================= Main =================
raw = load_repo_csv(override_path, skiprows=skiprows)
raw.columns = [str(c).strip() for c in raw.columns]

df = raw.loc[:, ~raw.columns.str.match(r"^Unnamed:\s*\d+$")]
df = df.dropna(axis=1, how="all")

# Identify key columns
col_rank       = resolve(df, "Rank")
col_visitors   = resolve(df, "Visitors", "VISITORS")
col_purchases  = resolve(df, "Purchasers", "Purchases", "BUYERS")
col_conversion = resolve(
    df,
    "Conversion %", "Conversion", "CONVERSION %", "CONVERSION",
    "Conversion%", "conversion%", "Conv", "CVR", "conversion_rate", "Conversion Rate"
)
col_depth      = resolve(df, "Depth")

attr_map = find_attributes(df)
attr_cols = [attr_map[k] for k in attr_map]

reserved = set([c for c in [col_rank, col_visitors, col_purchases, col_conversion, col_depth] if c]) | set(attr_cols)
sku_cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]

# ================= Filters =================
with st.expander("🔎 Filters", expanded=True):
    dff = df.copy()
    for label in ["Gender", "Credit rating"]:
        col = attr_map.get(label)
        if col:
            dff.loc[dff[col].astype(str).str.upper().str.strip() == "U", col] = pd.NA

    selections = {}
    exclude_flags = {}
    if attr_cols:
        st.markdown("**Attributes**")
        cols = st.columns(3)
        for i, (label, col) in enumerate(attr_map.items()):
            with cols[i % 3]:
                exclude_flags[label] = st.checkbox(f"Do not include {label}", value=False, key=f"ex_{label}")
                if not exclude_flags[label]:
                    vals = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    pick = st.multiselect(label, options=vals, default=[], key=f"ms_{label}")
                    if pick:
                        selections[col] = pick
        for col, vals in selections.items():
            dff = dff[dff[col].isin(vals)]
        for label, do_exclude in exclude_flags.items():
            if do_exclude:
                col = attr_map[label]
                if col in dff.columns:
                    dff = dff[
                        dff[col].isna() |
                        (dff[col].astype(str).str.strip() == "")
                    ]
    if col_visitors:
        dff[col_visitors] = pd.to_numeric(dff[col_visitors], errors="coerce")
        dff = dff[dff[col_visitors] >= int(min_rows)]
    st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# ================= Sorting & Ranking =================
if col_conversion is None and col_visitors and col_purchases:
    computed_conv_col = "__conv"
    dff[computed_conv_col] = 100.0 * pd.to_numeric(dff[col_purchases], errors="coerce") / \
                             pd.to_numeric(dff[col_visitors], errors="coerce").replace(0, np.nan)
    col_conversion = computed_conv_col

sort_map = {"Conversion": col_conversion, "Purchasers": col_purchases, "Visitors": col_visitors}
sort_col = sort_map.get(metric_choice)

if sort_col is None:
    if col_purchases is not None:
        st.warning("‘Conversion’ not found. Sorting by Purchasers instead.")
        metric_choice = "Purchasers"
        sort_col = col_purchases
    elif col_visitors is not None:
        st.warning("‘Conversion’ not found. Sorting by Visitors instead.")
        metric_choice = "Visitors"
        sort_col = col_visitors
    else:
        st.error("Couldn’t find usable sort columns (Conversion, Purchasers, or Visitors). Please add them to the CSV.")
        st.stop()

dff = dff.sort_values(sort_col, ascending=False, na_position="last").head(top_n).reset_index(drop=True)

if "Rank" in dff.columns:
    dff["Rank"] = np.arange(1, len(dff) + 1)
else:
    dff.insert(0, "Rank", np.arange(1, len(dff) + 1))

# ================= Formatting =================
if col_visitors:
    dff["Visitors_fmt"] = fmt_int_series(pd.to_numeric(dff[col_visitors], errors="coerce"))
if col_purchases:
    dff["Purchasers_fmt"] = fmt_int_series(pd.to_numeric(dff[col_purchases], errors="coerce"))
if col_depth and col_depth in dff.columns:
    dff["Depth_fmt"] = fmt_int_series(pd.to_numeric(dff[col_depth], errors="coerce"))

if col_conversion:
    if looks_like_percent_strings(dff[col_conversion]):
        dff["Conversion_fmt"] = dff[col_conversion].astype(str)
    else:
        dff["Conversion_fmt"] = pd.to_numeric(dff[col_conversion], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}%"
        )
else:
    dff["Conversion_fmt"] = ""

for sc in sku_cols:
    dff[sc] = fmt_int_series(pd.to_numeric(dff[sc], errors="coerce"))

# ================= Column order =================
attr_order_labels = [
    "Gender", "Age", "Homeowner", "Married", "Children",
    "Credit rating", "Income", "Net worth", "State",
    "Ethnicity (skiptrace)", "Department", "Seniority level", "Skiptrace credit"
]

excluded_labels = {lbl for lbl, flag in exclude_flags.items() if flag} if "exclude_flags" in locals() else set()
visible_attr_labels = [lbl for lbl in attr_order_labels if lbl in attr_map and lbl not in excluded_labels]
ordered_attr_cols = [attr_map[lbl] for lbl in visible_attr_labels]

table_cols = ["Rank"]
if col_visitors:   table_cols.append("Visitors_fmt")
if col_purchases:  table_cols.append("Purchasers_fmt")
table_cols.append("Conversion_fmt")
table_cols += ordered_attr_cols
table_cols += sku_cols
if col_depth and "Depth_fmt" in dff.columns:
    table_cols.append("Depth_fmt")

rename_map = {
    "Visitors_fmt": "Visitors",
    "Purchasers_fmt": "Purchasers",
    "Conversion_fmt": "Conversion",
}
if "Depth_fmt" in table_cols:
    rename_map["Depth_fmt"] = "Depth"

disp = dff[table_cols].rename(columns=rename_map)

# ================= Show & Download =================
st.dataframe(disp, use_container_width=True, hide_index=True)
st.download_button(
    "Download ranked combinations (CSV)",
    data=disp.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations_precomputed.csv",
    mime="text/csv"
)

data=disp.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations_precomputed.csv",
    mime="text/csv"
)
