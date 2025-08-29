
import pandas as pd
import numpy as np

PURCHASE_GUESS_CANDIDATES = ["PURCHASE", "Purchase", "purchased", "purchasers", "OrderCount", "orders", "Buyer"]

def coerce_purchase_series(df: pd.DataFrame, purchase_col: str) -> pd.Series:
    s = df[purchase_col]
    if np.issubdtype(s.dtype, np.number):
        return (s.fillna(0) > 0).astype(int)
    s_str = s.astype(str).str.strip().str.lower()
    true_vals = {"1","true","yes","y","t","buyer","purchased"}
    return s_str.isin(true_vals).astype(int)

def pick_default_purchase_col(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    for guess in PURCHASE_GUESS_CANDIDATES:
        if guess in cols:
            return guess
    for c in df.columns:
        lc = c.lower()
        if "order" in lc or "purchase" in lc or "buyer" in lc:
            return c
    return None

def safe_percent(numer: int, denom: int) -> float:
    return 0.0 if denom == 0 else (numer / denom) * 100.0

def detect_date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in {"date", "orderdate", "firstorderdate", "lastorderdate", "created_at"}:
            return c
    return None

def coerce_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    try:
        return pd.to_datetime(df[col], errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(df)))
