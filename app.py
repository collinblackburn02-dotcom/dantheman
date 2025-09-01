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
    st.write("CSV loaded successfully with", len(df), "rows.")
except FileNotFoundError:
    st.error("CSV file 'Copy of DAN_HHS - Sample.csv' not found! Please check the file path or upload the file to the app directory.")
    st.stop()

# Initialize DuckDB
con = duckdb.connect()
con.register('customers', df)

# Streamlit UI
st.title("Ranked Customer Dashboard")

# Minimum visitors input
min_visitors = st.number_input(
    "Minimum number of visitors to show a group",
    min_value=0,
    value=10,  # Default to 10 for testing (dataset has 197 rows)
    step=1,
    help="Your dataset has 197 rows, so groups can't exceed 197 visitors. Default 400 is for larger datasets."
)

# Available attributes (mapped to CSV columns)
available_attributes = [
    'Gender', 'Age_Range', 'Income_Range', 'Net_Worth', 'Home_Owner', 'Married',
    'Children', 'Credit_Rating', 'State'
]
# Map display names to CSV column names (handle aliases where needed, like STATE)
column_mapping = {
    'Gender': 'Gender',
    'Age_Range': 'Age Range',
    'Income_Range': 'Income Range',
    'Net_Worth': 'New Worth',
    'Home_Owner': 'Home Owner',
    'Married': 'Married',
    'Children': 'Children',
    'Credit_Rating': 'Credit Rating',
    'State': 'State'  # Could add aliases like 'PERSONAL_STATE' if needed
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
            include = st.checkbox(f"Include {attr}", value=True, key=f"checkbox_{attr}")
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
    
    # Debug: Show applied filters
    if where_clause:
        st.write("Applied filters:", where_clause)

    # Construct final SELECT
    select_clause = ", ".join([f"COALESCE({col}, '') AS {col}" for col in selected_attributes])
    
    # GROUP BY clause
    group_by_clause = ", ".join(selected_attributes)
    
    # Post-grouping filter for specific values
    post_filter = []
    for col, vals in specific_values.items():
        if vals:
            vals_str = ", ".join([f"'{v}'" for v in vals])
            post_filter.append(f"COALESCE({col}, '') IN ({vals_str})")
    
    post_filter_clause = " AND ".join(post_filter)
    if post_filter_clause:
        post_filter_clause = f"WHERE {post_filter_clause}"
    
    query = f"""
    WITH cleaned AS (
        SELECT 
            {cleaned_select},
            CASE WHEN Purchase = 1 THEN 1 ELSE 0 END AS is_purchaser,
            Revenue
        FROM customers
        {where_clause}
    ),
    grouped AS (
        SELECT 
            {group_by_clause},
            COUNT(*) AS visitors,
            SUM(is_purchaser) AS purchases,
            SUM(Revenue) AS total_revenue,
            ROUND(SUM(is_purchaser) * 1.0 / COUNT(*), 2) AS conversion_rate,
            ROW_NUMBER() OVER (PARTITION BY {group_by_clause} ORDER BY total_revenue DESC) AS rn
        FROM cleaned
        GROUP BY CUBE ({group_by_clause})
    )
    SELECT 
        {select_clause},
        visitors,
        purchases,
        total_revenue,
        conversion_rate,
        RANK() OVER (ORDER BY total_revenue DESC NULLS LAST) AS rank
    FROM grouped
    WHERE rn = 1
    {post_filter_clause}
    HAVING COUNT(*) >= {min_visitors}
    ORDER BY total_revenue DESC NULLS LAST
    """
    
    # Execute and display
    try:
        st.write("Executing query:", query)  # Debug: Show the query
        result = con.execute(query).fetchdf()
        if result.empty:
            st.warning(f"No groups meet the minimum visitor threshold of {min_visitors}. Try a lower value (dataset has {len(df)} rows).")
        else:
            st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Query error: {e}")
        st.write("Debug: Selected attributes:", selected_attributes)
        st.write("Specific values:", specific_values)
        st.write("Data sample:", df.head().to_string())  # Show first few rows for context
