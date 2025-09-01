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

# Streamlit UI
st.title("Expanded Ranked Customer Dashboard")

# Minimum visitors input
min_visitors = st.number_input("Minimum number of visitors to show a group", min_value=0, value=400, step=1)

# Available attributes
available_attributes = ['Age_Range', 'Gender', 'Home_Owner', 'Net_Worth', 'Income_Range', 'State', 'Credit_Rating']

# Map display names to CSV column names
column_mapping = {
    'Age_Range': 'Age Range',
    'Gender': 'Gender',
    'Home_Owner': 'Home Owner',
    'Net_Worth': 'New Worth',
    'Income_Range': 'Income Range',
    'State': 'State',
    'Credit_Rating': 'Credit Rating'
}

# Select overall attributes
selected_attributes = st.multiselect(
    "Select attributes to group by",
    options=available_attributes,
    default=['Age_Range', 'Gender', 'Home_Owner']
)

# For each selected attribute, select specific values (multiple allowed)
specific_values = {}
for col in selected_attributes:
    # Get unique non-null values for the column
    unique_query = f"""
    SELECT DISTINCT "{column_mapping[col]}" AS val
    FROM customers
    WHERE "{column_mapping[col]}" IS NOT NULL AND "{column_mapping[col]}" != ''
    ORDER BY val
    """
    unique_df = con.execute(unique_query).fetchdf()
    unique_list = unique_df['val'].tolist()
    
    specific_values[col] = st.multiselect(
        f"Select specific values for {col} (leave empty for all)",
        options=unique_list,
        default=[]  # Default empty means all
    )

# Build dynamic query
if not selected_attributes:
    st.warning("Please select at least one attribute to display the table.")
else:
    # Construct cleaned SELECT
    cleaned_select = ", ".join([
        f'CASE WHEN "{column_mapping[col]}" = \'\' OR "{column_mapping[col]}" IS NULL THEN NULL ELSE "{column_mapping[col]}" END AS {col}'
        for col in selected_attributes
    ])
    
    # Construct WHERE clause for specific values
    where_clauses = []
    for col, vals in specific_values.items():
        if vals:  # If specific values selected, filter
            vals_str = ", ".join([f"'{v}'" for v in vals])
            where_clauses.append(f"{col} IN ({vals_str})")
    
    where_clause = " AND ".join(where_clauses)
    if where_clause:
        where_clause = f"WHERE {where_clause}"
    
    # Construct final SELECT
    select_clause = ", ".join([f"COALESCE({col}, '') AS {col}" for col in selected_attributes])
    
    # GROUP BY clause
    group_by_clause = ", ".join(selected_attributes)
    
    query = f"""
    WITH cleaned AS (
        SELECT 
            {cleaned_select},
            Purchase,
            Revenue
        FROM customers
        {where_clause}
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
    HAVING COUNT(*) >= {min_visitors}
    ORDER BY total_revenue DESC NULLS LAST
    """
    
    # Execute and display
    try:
        result = con.execute(query).fetchdf()
        st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Query error: {e}")
