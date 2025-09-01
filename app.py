import duckdb
import streamlit as st
import pandas as pd
import os

# Debug: Print current directory and files
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Load your CSV (replace with your actual file path, e.g., 'data/Copy of DAN_HHS - Sample.csv')
try:
    df = pd.read_csv('Copy of DAN_HHS - Sample.csv')
except FileNotFoundError:
    st.error("CSV file 'Copy of DAN_HHS - Sample.csv' not found! Please check the file path or upload the file to the app directory.")
    st.stop()

# Initialize DuckDB
con = duckdb.connect()
con.register('customers', df)

# Streamlit UI for selecting attributes
st.title("Expanded Ranked Customer Dashboard")
st.write("Select attributes to group by:")
available_attributes = ['Age_Range', 'Gender', 'Home_Owner', 'Net_Worth', 'Income_Range', 'State']
selected_attributes = st.multiselect(
    "Attributes",
    options=available_attributes,
    default=['Age_Range', 'Gender', 'Home_Owner']  # Default to original three
)

# Map display names to CSV column names
column_mapping = {
    'Age_Range': 'Age Range',
    'Gender': 'Gender',
    'Home_Owner': 'Home Owner',
    'Net_Worth': 'New Worth',
    'Income_Range': 'Income Range',
    'State': 'State'
}

# Build dynamic query based on selected attributes
if not selected_attributes:
    st.warning("Please select at least one attribute to display the table.")
else:
    # Construct SELECT and GROUP BY clauses
    select_clause = ", ".join([f"COALESCE({col}, '') AS {col}" for col in selected_attributes])
    cleaned_select = ", ".join([
        f"CASE WHEN \"{column_mapping[col]}\" = '' OR \"{column_mapping[col]}\" IS NULL THEN NULL ELSE \"{column_mapping[col]}\" END AS {col}"
        for col in selected_attributes
    ])
    group_by_clause = ", ".join(selected_attributes)

    query = f"""
    WITH cleaned AS (
        SELECT 
            {cleaned_select},
            Purchase,
            Revenue
        FROM customers
    )
    SELECT 
        {select_clause},
        COUNT(*) AS visitors,
        SUM(Purchase) AS purchases,
        SUM(Revenue) AS total_revenue,
        ROUND(SUM(Purchase) * 1.0 / COUNT(*), 2) AS conversion_rate,
        RANK() OVER (ORDER BY total_revenue DESC NULLS LAST) AS rank
    FROM cleaned
    GROUP BY CUBE ({group_by_clause})
    ORDER BY total_revenue DESC NULLS LAST
    """

    # Execute and display
    result = con.execute(query).fetchdf()
    st.dataframe(result, use_container_width=True)
# Execute and display
result = con.execute(query).fetchdf()
st.title("Expanded Ranked Customer Dashboard")
st.dataframe(result)
