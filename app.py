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
min_visitors = st.number_input(
    "Minimum number of visitors to show a group",
    min_value=0,
    value=10,  # Set to 10 for testing (dataset has 197 rows)
    step=1
)

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

# Select attributes with toggles, 3 per row
st.write("Select attributes to include:")
selected_attributes = []
specific_values = {}

num_cols = 3
rows = [available_attributes[i:i+num_cols] for i in range(0, len(available_attributes), num_cols)]

for row in rows:
    cols = st.columns(num_cols)
    for idx, attr in enumerate(row):
        with cols[idx]:
            include = st.checkbox(f"Include {attr}", value=attr in ['Age_Range', 'Gender', 'Home_Owner'], key=f"checkbox_{attr}")
            if include:
                selected_attributes.append(attr)
                unique_query = f"""
                SELECT DISTINCT "{column_mapping[attr]}" AS val
                FROM customers
                WHERE "{column_mapping[attr]}" IS NOT NULL AND "{column_mapping[attr]}" != ''
                ORDER BY val
                """
                unique_df = con.execute(unique_query).fetchdf()
                unique_list = unique_df['val'].tolist()
                specific_values[attr] = st.multiselect(
                    f"Values for {attr} (all if empty)",
                    options=unique_list,
                    default=[],
                    key=f"multiselect_{attr}"
                )

# Build dynamic query
if not selected_attributes:
    st.warning("Please include at least one attribute to display the table.")
else:
    # Construct cleaned SELECT
    cleaned_select = ", ".join([
        f'CASE WHEN "{column_mapping[col]}" = \'\' OR "{column_mapping[col]}" IS NULL THEN NULL ELSE "{column_mapping[col]}" END AS {col}'
        for col in selected_attributes
    ])
    
    # Construct WHERE clause for specific values (initial filter)
    where_clauses = []
    for col, vals in specific_values.items():
        if vals:
            vals_str = ", ".join([f"'{v}'" for v in vals])
            where_clauses.append(f'"{column_mapping[col]}" IN ({vals_str})')
    
    where_clause = " AND ".join(where_clauses)
    if where_clause:
        where_clause = f"WHERE {where_clause}"
    
    # Construct final SELECT
    select_clause = ", ".join([f"COALESCE({col}, '') AS {col}" for col in selected_attributes])
    
    # Determine GROUP BY type
    # If all selected attributes have specific values, use regular GROUP BY
    # Otherwise, use CUBE for attributes without specific values
    cube_attrs = [col for col in selected_attributes if not specific_values[col]]
    group_by_clause = ", ".join(selected_attributes)
    group_by_type = "CUBE" if cube_attrs else "GROUP BY"
    
    query = f"""
    WITH cleaned AS (
        SELECT 
            {cleaned_select},
            Purchase,
            Revenue
        FROM customers
        {where_clause}
    ),
    grouped AS (
        SELECT 
            {group_by_clause},
            COUNT(*) AS visitors,
            SUM(Purchase) AS purchases,
            SUM(Revenue) AS total_revenue,
            ROUND(SUM(Purchase) * 1.0 / COUNT(*), 2) AS conversion_rate
        FROM cleaned
        {group_by_type} ({group_by_clause})
        HAVING COUNT(*) >= {min_visitors}
    )
    SELECT 
        {select_clause},
        visitors,
        purchases,
        total_revenue,
        conversion_rate,
        RANK() OVER (ORDER BY total_revenue DESC NULLS LAST) AS rank
    FROM grouped
    WHERE {' AND '.join([f"{col} IN ({', '.join([f"'{v}'" for v in vals])})" for col, vals in specific_values.items() if vals]) or 'TRUE'}
    ORDER BY total_revenue DESC NULLS LAST
    """
    
    # Execute and display
    try:
        result = con.execute(query).fetchdf()
        if result.empty:
            st.warning(f"No groups meet the minimum visitor threshold of {min_visitors}.")
        else:
            st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Query error: {e}")
