
import pandas as pd
ALIASES = {
    "EMAIL": ["EMAIL","Email","email"],
    "PURCHASE": ["PURCHASE","Purchase","purchased","Buyer","is_buyer"],
    "DATE": ["DATE","Date","date","Last Order Date","LAST ORDER DATE","LastOrderDate"],
    "AGE_RANGE": ["AGE_RANGE","Age Range","age_range"],
    "CHILDREN": ["CHILDREN","Children"],
    "GENDER": ["GENDER","Gender"],
    "HOMEOWNER": ["HOMEOWNER","Homeowner"],
    "MARRIED": ["MARRIED","Married"],
    "NET_WORTH": ["NET_WORTH","Net Worth","NetWorth"],
    "INCOME_RANGE": ["INCOME_RANGE","Income Range","income_range"],
    "CREDIT_RATING": ["SKIPTRACE_CREDIT_RATING","Credit Rating","credit_rating","SKIPTRACE CREDIT RATING"],
    "MOST_RECENT_SKU": ["MostRecentSKU","Most Recent SKU","Recent SKU","SKU"],
    "PERSONAL_STATE": ["PERSONAL_STATE","State","STATE"],
    "REVENUE": ["Total","TOTAL","Order Total","ORDER TOTAL","Sale Price","Lineitem price","LINEITEM PRICE"],
}
def resolve_col(df: pd.DataFrame, key: str) -> str | None:
    cands = ALIASES.get(key, [])
    for c in cands:
        if c in df.columns: return c
        for dc in df.columns:
            if str(dc).strip().lower() == str(c).strip().lower():
                return dc
    return None
