import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ================= Branding =================
# Brand colors
BRAND = {
    "bg": "#F5F0E6",       # beige
    "fg": "#3A2A26",       # deep brown text
    "accent": "#6E4F3A",   # medium brown
    "accent2": "#A07A5A",  # lighter brown
    "card": "#FFF9F0",     # light card beige
    "white": "#FFFFFF",
}

# Try to find a logo automatically
LOGO_CANDIDATES = ["logo.png", "heavenly_logo.png", "assets/logo.png", "assets/heavenly_logo.png"]

def find_logo():
    base = Path.cwd()
    for name in LOGO_CANDIDATES:
        p = base / name
        if p.exists():
            return str(p)
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
            /* App background & global text */
            .stApp {{
                background: var(--bg);
                color: var(--fg);
            }}
            /* Main title & caption */
            .heavenly-title {{
                font-weight: 800;
                letter-spacing: 0.3px;
                color: var(--fg);
                margin: 0 0 .25rem 0;
            }}
            .heavenly-caption {{
                color: rgba(58,42,38,0.85);
                font-size: 0.95rem;
                margin-bottom: 1.25rem;
            }}

            /* Card look for containers */
            .block-container {{
                padding-top: 1.2rem;
            }}
            .stExpander, .stDataFrame, .stMarkdown, .stDownloadButton {{
                background: var(--card);
                border-radius: 14px;
            }}

            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--card), var(--bg));
                color: var(--fg);
                border-right: 1px solid rgba(58,42,38,0.08);
            }}
            /* Sidebar titles */
            section[data-testid="stSidebar"] .stMarkdown p {{
                color: var(--fg);
                font-weight: 700;
            }}

            /* Inputs */
            .stSlider > div > div > div > div {{
                background: var(--accent) !important;
            }}
            .stSlider [data-baseweb="slider"] > div:first-child {{
                background: rgba(110,79,58,0.25) !important;
            }}
            .stNumberInput input, .stTextInput input {{
                background: var(--white);
                color: var(--fg);
                border: 1px solid rgba(110,79,58,0.25);
                border-radius: 10px;
            }}
            /* Multiselect chips */
            div[data-baseweb="tag"] {{
                background: rgba(110,79,58,0.15);
                color: var(--fg);
            }}

            /* Radio/Select styling */
            .stRadio label, .stSelectbox label {{
                color: var(--fg) !important;
                font-weight: 600;
            }}

            /* Buttons */
            .stDownloadButton button, .stButton button {{
                background: var(--accent) !important;
                color: var(--white) !important;
                border-radius: 12px !important;
                border: 1px solid rgba(58,42,38,0.15) !important;
                box-shadow: 0 2px 12px rgba(58,42,38,0.12);
            }}
            .stDownloadButton button:hover, .stButton button:hover {{
                background: var(--accent2) !important;
            }}

            /* Expander header */
            details > summary {{
                background: var(--card);
                border-radius: 12px;
                padding: .6rem .9rem;
                border: 1px solid rgba(58,42,38,0.08);
                color: var(--fg);
                font-weight: 700;
            }}

            /* Dataframe tweaks */
            div[data-testid="stDataFrame"] {{
                border: 1px solid rgba(58,42,38,0.08);
                box-shadow: 0 2px 16px rgba(58,42,38,0.08);
                border-radius: 12px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ================= App Meta =================
st.set_page_config(page_title="Ranked Customer Dashboard — Precomputed CSV", layout="wide")
inject_css()

# Header with logo
logo_path = find_logo()
col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
with col_logo:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with col_title:
    st.markdown('<h1 class="heavenly-title">Heavenly Health / Heavenly Heat — Ranked Customer Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="heavenly-caption">Loads your Sheets export (skip first 3 rows), then filters, ranks, and displays. No recomputation.</div>', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Controls")
    uploaded = st.file_uploader("Upload precomputed CSV", type=["csv"])
    st.markdown("---")
    metric_choice = st.radio("Sort metric", ["Conversion", "Purchasers", "Visitors"], index=0)
    top_n = st.slider("Top N", 10, 2000, 50, 10)
    # Min visitors: floor 100, default 100
    min_rows = st.number_input("Minimum Visitors per group", min_value=100, value=100, step=1)

# ---------------- Helpers ----------------
def resolve(df: pd.DataFrame, *candidates):
    """Return the first matching column (case-insensitive), else None."""
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = str(cand).strip().lower()
        if k in norm:
            return norm[k]
    return None

def find_attributes(df: pd.DataFrame):
    """
    Identify attribute columns by typical names; returns {friendly_label: actual_col}.
    (We keep friendly labels for the UI; the table uses your exact CSV headers.)
    """
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

# ---------------- Main ----------------
if not uploaded:
    st.info("Upload the precomputed CSV (the one you exported from Sheets) to begin.")
    st.stop()

# Load CSV (skip first 3 rows for real header)
raw = pd.read_csv(uploaded, skiprows=3)
raw.columns = [str(c).strip() for c in raw.columns]

# Drop unnamed or all-empty columns
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

# SKU columns = numeric columns not in metrics/attributes/rank/depth
reserved = set([c for c in [col_rank, col_visitors, col_purchases, col_conversion, col_depth] if c]) | set(attr_cols)
sku_cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]

# ---------------- Filters ----------------
with st.expander("🔎 Filters", expanded=True):
    dff = df.copy()

    # Optionally treat 'U' as missing for these attributes
    for label in ["Gender", "Credit rating"]:
        col = attr_map.get(label)
        if col:
            dff.loc[dff[col].astype(str).str.upper().str.strip() == "U", col] = pd.NA

    selections = {}
    exclude_flags = {}  # label -> True/False ("Do not include")
    if attr_cols:
        st.markdown("**Attributes**")
        cols = st.columns(3)
        for i, (label, col) in enumerate(attr_map.items()):
            with cols[i % 3]:
                # Do not include toggle
                exclude_flags[label] = st.checkbox(f"Do not include {label}", value=False, key=f"ex_{label}")

                # If NOT excluded, allow picking specific values to include
                if not exclude_flags[label]:
                    vals = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                    pick = st.multiselect(label, options=vals, default=[], key=f"ms_{label}")
                    if pick:
                        selections[col] = pick

        # Apply value selections (only for included attributes)
        for col, vals in selections.items():
            dff = dff[dff[col].isin(vals)]

        # Apply "Do not include" filters -> keep rows where the excluded attribute is blank/NA
        for label, do_exclude in exclude_flags.items():
            if do_exclude:
                col = attr_map[label]
                if col in dff.columns:
                    dff = dff[
                        dff[col].isna() |
                        (dff[col].astype(str).str.strip() == "")
                    ]

    # Enforce min Visitors if present
    if col_visitors:
        dff[col_visitors] = pd.to_numeric(dff[col_visitors], errors="coerce")
        dff = dff[dff[col_visitors] >= int(min_rows)]

    st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# ---------------- Sorting & Ranking ----------------
computed_conv_col = None
if col_conversion is None and col_visitors and col_purchases:
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

# Safe rank: overwrite if exists, else insert
if "Rank" in dff.columns:
    dff["Rank"] = np.arange(1, len(dff) + 1)
else:
    dff.insert(0, "Rank", np.arange(1, len(dff) + 1))

# ---------------- Formatting ----------------
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

# ---------------- Column order ----------------
# Add the new attribute labels to the display order
attr_order_labels = [
    "Gender", "Age", "Homeowner", "Married", "Children",
    "Credit rating", "Income", "Net worth", "State",
    "Ethnicity (skiptrace)", "Department", "Seniority level", "Skiptrace credit"
]

# Hide any attributes the user marked "Do not include"
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

# ---------------- Show & Download ----------------
st.dataframe(disp, use_container_width=True, hide_index=True)

st.download_button(
    "Download ranked combinations (CSV)",
    data=disp.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations_precomputed.csv",
    mime="text/csv"
)
