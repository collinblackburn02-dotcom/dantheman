import duckdb
import streamlit as st
import pandas as pd

# Load your CSV (replace with your file path)
df = pd.read_csv('Copy of DAN_HHS - Sample.csv')

# Initialize DuckDB
con = duckdb.connect()
con.register('customers', df)

# Query with CUBE on all attributes, using empty string for rollup groups
query = """
WITH cleaned AS (
    SELECT 
        CASE WHEN "Age Range" = '' OR "Age Range" IS NULL THEN NULL ELSE "Age Range" END AS Age_Range,
        CASE WHEN Gender = '' OR Gender IS NULL THEN NULL ELSE Gender END AS Gender,
        CASE WHEN "Home Owner" = '' OR "Home Owner" IS NULL THEN NULL ELSE "Home Owner" END AS Home_Owner,
        CASE WHEN "New Worth" = '' OR "New Worth" IS NULL THEN NULL ELSE "New Worth" END AS Net_Worth,
        CASE WHEN "Income Range" = '' OR "Income Range" IS NULL THEN NULL ELSE "Income Range" END AS Income_Range,
        CASE WHEN State = '' OR State IS NULL THEN NULL ELSE State END AS State,
        Purchase,
        Revenue
    FROM customers
)
SELECT 
    COALESCE(Age_Range, '') AS Age_Range,
    COALESCE(Gender, '') AS Gender,
    COALESCE(Home_Owner, '') AS Home_Owner,
    COALESCE(Net_Worth, '') AS Net_Worth,
    COALESCE(Income_Range, '') AS Income_Range,
    COALESCE(State, '') AS State,
    COUNT(*) AS visitors,
    SUM(Purchase) AS purchases,
    SUM(Revenue) AS total_revenue,
    ROUND(SUM(Purchase) * 1.0 / COUNT(*), 2) AS conversion_rate,
    RANK() OVER (ORDER BY total_revenue DESC NULLS LAST) AS rank
FROM cleaned
GROUP BY CUBE (Age_Range, Gender, Home_Owner, Net_Worth, Income_Range, State)
ORDER BY total_revenue DESC NULLS LAST
"""

# Execute and display
result = con.execute(query).fetchdf()
st.title("Expanded Ranked Customer Dashboard")
st.dataframe(result)
