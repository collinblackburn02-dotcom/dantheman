
import pandas as pd
import numpy as np

# Candidate column names the app will accept (system or human-friendly)
ALIASES = {
    "EMAIL": ["EMAIL","Email","email"],
    "PURCHASE": ["PURCHASE","Purchase","purchased","Buyer","is_buyer"],
    "DATE": ["DATE","Date","date"],
    "AGE_RANGE": ["AGE_RANGE","Age Range","age_range"],
    "CHILDREN": ["CHILDREN","Children"],
    "GENDER": ["GENDER","Gender"],
    "HOMEOWNER": ["HOMEOWNER","Homeowner"],
    "MARRIED": ["MARRIED","Married"],
    "NET_WORTH": ["NET_WORTH","Net Worth","NetWorth"],
    "INCOME_RANGE": ["INCOME_RANGE","Income Range","income_range"],
    "CREDIT_RATING": ["SKIPTRACE_CREDIT_RATING","Credit Rating","credit_rating","SKIPTRACE CREDIT RATING"],
    "EXACT_AGE": ["SKIPTRACE_EXACT_AGE","Exact Age","SKIPTRACE EXACT AGE"],
    "ETHNIC_CODE": ["SKIPTRACE_ETHNIC_CODE","Ethnic Code","SKIPTRACE ETHNIC CODE"],
    "ORDER_COUNT": ["OrderCount","Order Count","Orders","orders"],
    "FIRST_ORDER_DATE": ["FirstOrderDate","First Order Date"],
    "LAST_ORDER_DATE": ["LastOrderDate","Last Order Date"],
    "REVENUE": ["Revenue","Total Revenue","Total"],
    "SKUS": ["SKUs","Sku List","SKU List"],
    "MOST_RECENT_SKU": ["MostRecentSKU","Most Recent SKU","Recent SKU"],
}

def resolve_col(df: pd.DataFrame, key: str) -> str | None:
    """Return the first column name present in df for the alias key."""
    cands = ALIASES.get(key, [])
    for c in cands:
        if c in df.columns:
            return c
        # allow loose match on stripped case
        for dc in df.columns:
            if str(dc).strip().lower() == str(c).strip().lower():
                return dc
    return None

def coerce_purchase_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    s2 = s.astype(str).str.strip().str.lower()
    yes = {"1","true","t","yes","y","buyer","purchased"}
    return s2.isin(yes).astype(int)

def safe_percent(n, d):
    return 0.0 if (d is None or d == 0) else (n/d)*100.0

def to_datetime_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(s)))
