import re

import pandas as pd
import numpy as np

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
    "ORDER_COUNT": ["OrderCount","Order Count","Orders","orders"],
    "FIRST_ORDER_DATE": ["FirstOrderDate","First Order Date"],
    "LAST_ORDER_DATE": ["LastOrderDate","Last Order Date"],
    "REVENUE": ["Revenue","Total Revenue","Total"],
    "SKUS": ["SKUs","Sku List","SKU List"],
    "MOST_RECENT_SKU": ["MostRecentSKU","Most Recent SKU","Recent SKU"],
    "ZIP": ["PERSONAL_ZIP","Billing Zip","Shipping Zip","Zip","ZIP","Postal Code"]
}

def resolve_col(df: pd.DataFrame, key: str) -> str | None:
    cands = ALIASES.get(key, [])
    for c in cands:
        if c in df.columns:
            return c
        for dc in df.columns:
            if str(dc).strip().lower() == str(c).strip().lower():
                return dc
    return None

def coerce_purchase(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    vals = s.astype(str).str.strip().str.lower()
    yes = {"1","true","t","yes","y","buyer","purchased"}
    return vals.isin(yes).astype(int)

def to_datetime_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(s)))

def safe_percent(n, d):
    return 0.0 if (d is None or d == 0) else (n/d)*100.0

def explode_skus(df: pd.DataFrame, skus_col: str, sep: str = ";"):
    # Returns rows duplicated per SKU value for purchasers (Purchase==1)
    d = df[df["_PURCHASE"] == 1].copy()
    d[skus_col] = d[skus_col].astype(str)
    d["__sku_list"] = d[skus_col].fillna("").apply(lambda x: [s.strip() for s in str(x).split(sep) if str(s).strip()])
    d = d.explode("__sku_list")
    d = d.rename(columns={"__sku_list": "__SKU"})
    d = d[d["__SKU"].notna() & (d["__SKU"] != "")]
    return d


def clean_sku_token(tok: str) -> str | None:
    if tok is None: 
        return None
    s = str(tok).strip()
    if not s: 
        return None
    # Drop tokens that contain spaces (likely city names / phrases)
    if " " in s:
        return None
    # Keep only reasonable SKU characters
    if re.fullmatch(r"[A-Za-z0-9_-]{2,40}", s) is None:
        return None
    return s
