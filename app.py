import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ================ Brand palette & CSS =================
BRAND = {
    "bg": "#F5F0E6",        # beige background
    "fg": "#3A2A26",        # deep brown text
    "accent": "#6E4F3A",    # medium brown
    "accent2": "#A07A5A",   # lighter brown
    "card": "#FFF9F0",      # light card beige
    "white": "#FFFFFF",
}

LOGO_CANDIDATES = [
    "logo.png",
    "heavenly_logo.png",
    "assets/logo.png",
    "assets/heavenly_logo.png",
]

def find_logo_path():
    base = Path.cwd()
    for p in LOGO_CANDIDATES:
        pp = base / p
        if pp.exists():
            return str(pp)
    return None

def inject_css():
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {BRAND["bg"]};
                --fg: {BRAND["fg"]};
                --accent: {BRAND["accent"]};
                --accent2: {BRAND["accent2"]};
                --card: {BRAND["card"]};
                --white: {BRAND["white"]};
            }}
            .stApp {{ background: var(--bg); color: var(--fg); }}
            .block-container {{ padding-top: 1rem; }}
            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--card), var(--bg));
                color: var(--fg);
                border-right: 1px solid rgba(58,42,38,0.08);
            }}
            section[data-testid="stSidebar"] .stMarkdown p {{
                color: var(--fg); font-weight: 700;
            }}
            /* Inputs */
            .stNumberInput input, .stTextInput input {{
                background: var(--white); color: var(--fg);
                border: 1px solid rgba(110,79,58,0.25); border-radius: 10px;
            }}
            .stSlider > div > div > div > div {{ background: var(--accent) !important; }}
            .stSlider [data-baseweb="slider"] > div:first-child {{ background: rgba(110,79,58,0.25) !important; }}
            .stRadio label, .stSelectbox label {{ color: var(--fg) !important; font-weight: 600; }}

            /* Buttons */
            .stDownloadButton button, .stButton button {{
                background: var(--accent) !important; color: var(--white) !important;
                border-radius: 12px !important; border: 1px solid rgba(58,42,38,0.15) !important;
                box-shadow: 0 2px 12px rgba(58,42,38,0.12);
            }}
            .stDownloadButton button:hover, .stButton button:hover {{ background: var(--accent2) !important; }}

            /* Cards / dataframes */
            div[data-testid="stDataFrame"] {{
                border: 1px solid rgba(58,42,38,0.08);
                box-shadow: 0 2px 16px rgba(58,42,38,0.08);
                border-radius: 12px;
                background: var(--card);
            }}
            .stExpander {{
                background: var(--card);
                border-radius: 14px;
            }}

            /* Section headers */
            .heavenly-section-title {{
                font-size: 1.6rem; font-weight: 800; color: var(--fg);
                margin: 1.5rem 0 1rem 0;
            }}
            .heavenly-attr-title {{
                font-size: 1.25rem; font-weight: 700; color: var(--fg);
                margin: 1.2rem 0 0.5rem 0;
            }}

            /* Inline label + exclude checkbox alignment */
            .inline-label {{
                display: flex; align-items: center; gap: .5rem;
                font-weight: 600; color: var(--fg);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ================ App meta & header =================
st.set_page_config(page_title="Ranked Customer Dashboard — Precomputed CSV", layout="wide")
inject_css()

# Centered logo only
logo = find_logo_path()
if logo:
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.markdown("<div style='padding-top:25px'></div>", unsafe_allow_html=True)
        st.image(logo, use_container_width=False)

# ================ Sidebar (controls) =================
with st.sidebar:
    st.markdown("### Controls")
    uploaded = st.file_uploader("Upload precomputed CSV", type=["csv"])
    st.markdown("---")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    # Minimum visitors: floor 100, default 100
    min_rows = st.number_input("Minimum Visitors per group", min_value=100, value=100, step=1)

# Fixed: Top N = 100 (hidden)
top_n = 100

# ================ Helpers =================
def resolve(df: pd.DataFrame, *candidates):
    """Return the first matching column (case-insensitive), else None."""
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = str(cand).strip().lower()
        if k in norm:
            return norm[k]
    return None

def find_attributes(df: pd.DataFrame):
    """Map friendly labels -> actual CSV columns (only those present)."""
    m = {
        "Gender":               resolve(df, "Gender", "GENDER"),
        "Age":                  resolve(df, "Age_Range", "AGE_RANGE", "Age", "AGE"),
        "Homeowner":            resolve(df, "Homeowner", "HOMEOWNER"),
        "Married":              resolve(df, "Married", "MARRIED"),
        "Children":             resolve(df, "Children", "CHILDREN"),
        "Credit rating":        resolve(df, "Credit_Rating", "CREDIT_RATING", "Credit", "CREDIT"),
        "Income":               resolve(df, "Income_Range", "INCOME_RANGE", "Income", "INCOME"),
        "Net worth":            resolve(df, "New_Worth", "NET_WORTH", "NETWORTH", "Net_Worth"),
        "State":                resolve(df, "State", "PERSONAL_STATE", "STATE"),
        # Extras
        "Ethnicity (skiptrace)": resolve(df, "SKIPTRACE_ETHNIC_CODE", "Skiptrace_Ethnic_Code"),
        "Department":            resolve(df, "DEPARTMENT", "Department"),
        "Seniority level":       resolve(df, "SENIORITY_LEVEL", "Seniority_Level", "Seniority"),
        "Skiptrace credit":      resolve(df, "SKIPTRACE_CREDIT_RATING", "Skiptrace_Credit_Rating"),
    }
    return {k: v for k, v in m.items() if v is not None}

def fmt_int_series(s: pd.Series) -> pd.Series:
    return s.apply(lambda v: "" if pd.isna(v) else f"{int(round(float(v))):,}")

def looks_like_percent_strings(s: pd.Series) -> bool:
    if s.dtype == "O":
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        return (sample.str.contains("%").mean() > 0.6)
    return False

# ================ Main: load CSV =================
if not uploaded:
    st.info("Upload the precomputed CSV (the one you exported from Sheets) to begin.")
    st.stop()

# Real header starts at row 4
raw = pd.read_csv(uploaded, skiprows=3)
raw.columns = [str(c).strip() for c in raw.columns]

# Clean frame
df = raw.loc[:, ~raw.columns.str.match(r"^Unnamed:\s*\d+$")]
df = df.dropna(axis=1, how="all")

# Identify columns
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

# SKU: numeric columns not used above
reserved = set([c for c in [col_rank, col_visitors, col_purchases, col_conversion, col_depth] if c]) | set(attr_cols)
sku_cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]

# ================ Filters (inline exclude + selector) ================
with st.expander("🔎 Filters", expanded=True):
    dff = df.copy()

    # Normalize 'U' to NA for a couple attrs
    for label in ["Gender", "Credit rating"]:
        c = attr_map.get(label)
        if c:
            dff.loc[dff[c].astype(str).str.upper().str.strip() == "U", c] = pd.NA

    selections = {}
    exclude_flags = {}

    if attr_cols:
        st.markdown("**Attributes**")
        cols = st.columns(3)
        for i, (label, col) in enumerate(attr_map.items()):
            with cols[i % 3]:
                left, right = st.columns([1, 2])
                with left:
                    # Inline exclude
                    exclude_flags[label] = st.checkbox(label, value=False, key=f"ex_{label}")
                with right:
                    # Values list (disabled when excluded)
                    vals = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    picked = st.multiselect(
                        label,
                        options=vals,
                        default=[],
                        disabled=exclude_flags[label],
                        key=f"ms_{label}",
                        label_visibility="collapsed",
                    )
                    if picked:
                        selections[col] = picked

        # Apply chosen values (for included attrs)
        for c, vals in selections.items():
            dff = dff[dff[c].isin(vals)]

        # Apply "Do not include": keep rows where excluded attr is blank
        for label, do_ex in exclude_flags.items():
            if do_ex:
                c = attr_map[label]
                if c in dff.columns:
                    dff = dff[dff[c].isna() | (dff[c].astype(str).str.strip() == "")]

    # Min visitors
    if col_visitors:
        dff[col_visitors] = pd.to_numeric(dff[col_visitors], errors="coerce")
        dff = dff[dff[col_visitors] >= int(min_rows)]

    st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# ================ Sort & rank =================
if col_conversion is None and col_visitors and col_purchases:
    # compute conversion if missing
    computed_conv_col = "__conv"
    dff[computed_conv_col] = 100.0 * pd.to_numeric(dff[col_purchases], errors="coerce") / \
                             pd.to_numeric(dff[col_visitors], errors="coerce").replace(0, np.nan)
    col_conversion = computed_conv_col

sort_map = {"Conversion": col_conversion, "Purchasers": col_purchases, "Visitors": col_visitors}
sort_col = sort_map[metric_choice]
if sort_col is None:
    st.error(f"Missing column required to sort by '{metric_choice}'. Please include it in your CSV.")
    st.stop()

dff = dff.sort_values(sort_col, ascending=False, na_position="last").head(top_n).reset_index(drop=True)

# Rank (1..N)
if "Rank" in dff.columns:
    dff["Rank"] = np.arange(1, len(dff) + 1)
else:
    dff.insert(0, "Rank", np.arange(1, len(dff) + 1))

# ================ Formatting for main table =================
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

# Column order
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

# Style: hide "None"/"nan" by blending to bg; bold Conversion
bg = BRAND["bg"]
def style_hide_none(v):
    s = str(v).strip().lower()
    if s in ("none", "nan", ""):
        return f"color: {bg}"
    return ""
def style_bold(v):
    return "font-weight: 700" if str(v).strip() != "" else ""

styler = disp.style.applymap(style_hide_none)
if "Conversion" in disp.columns:
    styler = styler.applymap(style_bold, subset=["Conversion"])

st.dataframe(styler, use_container_width=True, hide_index=True)

st.download_button(
    "Download ranked combinations (CSV)",
    data=disp.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations_precomputed.csv",
    mime="text/csv"
)

# ================ Single-Attribute Summary Tables (unfiltered) =================
# Build from ORIGINAL df (no interaction), depth-1 only, rank by Conversion desc.
def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")

def _safe_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _singleton_summary_table(df_all: pd.DataFrame, attr_label: str, attr_col: str, drop_values: set[str] | None = None) -> pd.DataFrame:
    active_attr_cols = [c for c in attr_cols if c in df_all.columns]
    other_cols = [c for c in active_attr_cols if c != attr_col]

    has_attr = ~_is_blank(df_all[attr_col])
    others_blank = pd.Series(True, index=df_all.index)
    for oc in other_cols:
        others_blank &= _is_blank(df_all[oc])

    base = df_all[has_attr & others_blank].copy()
    if base.empty:
        return pd.DataFrame(columns=[attr_label, "Visitors", "Purchasers", "Conversion"])

    if drop_values:
        keep_mask = ~base[attr_col].astype(str).str.strip().str.upper().isin({v.upper() for v in drop_values})
        base = base[keep_mask]
        if base.empty:
            return pd.DataFrame(columns=[attr_label, "Visitors", "Purchasers", "Conversion"])

    vis_col = col_visitors if col_visitors in base.columns else resolve(base, "Visitors")
    pur_col = col_purchases if col_purchases in base.columns else resolve(base, "Purchasers")

    base[vis_col] = _safe_number(base[vis_col])
    base[pur_col] = _safe_number(base[pur_col])

    agg = base.groupby(attr_col, dropna=False).agg(
        Visitors=(vis_col, "sum"),
        Purchasers=(pur_col, "sum"),
    ).reset_index()

    agg["__conv_num"] = 100.0 * agg["Purchasers"] / agg["Visitors"].replace(0, np.nan)
    agg = agg.sort_values("__conv_num", ascending=False, na_position="last")

    agg["Visitors"] = agg["Visitors"].map(lambda x: "" if pd.isna(x) else f"{int(round(float(x))):,}")
    agg["Purchasers"] = agg["Purchasers"].map(lambda x: "" if pd.isna(x) else f"{int(round(float(x))):,}")
    agg["Conversion"] = agg["__conv_num"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    agg = agg.drop(columns="__conv_num").rename(columns={attr_col: attr_label})
    return agg

st.markdown("---")
st.markdown('<div class="heavenly-section-title">📑 Single-Attribute Summary Tables (unfiltered)</div>', unsafe_allow_html=True)

summary_order = [
    "Gender", "Age", "Homeowner", "Married", "Children",
    "Credit rating", "Income", "Net worth", "State",
    "Ethnicity (skiptrace)", "Department", "Seniority level", "Skiptrace credit",
]

for label in summary_order:
    if label not in attr_map:
        continue
    col = attr_map[label]
    if col not in df.columns:
        continue

    drop_vals = {"U"} if label == "Gender" else None
    tbl = _singleton_summary_table(df, label, col, drop_values=drop_vals)

    st.markdown(f'<div class="heavenly-attr-title">{label}</div>', unsafe_allow_html=True)
    if tbl.empty:
        st.caption("No single-attribute groups found for this attribute in the file.")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True)
