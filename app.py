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
st.title("Ranked Customer Dashboard")

min_visitors = st.number_input(
    "Minimum number of visitors to show a group",
    min_value=0,
    value=400,
    step=1
)

available_attributes = ['Age_Range', 'Gender', 'Home_Owner', 'Net_Worth', 'Income_Range', 'State', 'Credit_Rating']

column_mapping = {
    'Age_Range': 'Age Range',
    'Gender': 'Gender',
    'Home_Owner': 'Home Owner',
    'Net_Worth': 'New Worth',
    'Income_Range': 'Income Range',
    'State': 'State',
    'Credit_Rating': 'Credit Rating'
}

st.write("Select attributes to include:")

selected_attributes = []
specific_values = {}

num_cols = 3
rows = [available_attributes[i:i+num_cols] for i in range(0, len(available_attributes), num_cols)]

for row in rows:
    cols = st.columns(num_cols)
    for idx, attr in enumerate(row):
        with cols[idx]:
            include = st.checkbox(f"Include {attr}", value=attr in ['Age_Range', 'Gender', 'Home_Owner'])
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
                    default=[]
                )

if not selected_attributes:
    st.warning("Please include at least one attribute to display the table.")
else:
    cleaned_select = ", ".join([
        f'CASE WHEN "{column_mapping[col]}" = \'\' OR "{column_mapping[col]}" IS NULL THEN NULL ELSE "{column_mapping[col]}" END AS {col}'
        for col in selected_attributes
    ])
    
    where_clauses = []
    for col, vals in specific_values.items():
        if vals:
            vals_str = ", ".join([f"'{v.replace(\"'\", \"\\\\'\")}'" for v in vals])
            where_clauses.append(f"{col} IN ({vals_str})")
    
    where_clause = " AND ".join(where_clauses)
    if where_clause:
        where_clause = f"WHERE {where_clause}"
    
    select_clause = ", ".join([f"COALESCE({col}, '') AS {col}" for col in selected_attributes])
    
    filtered_attrs = [col for col in selected_attributes if specific_values[col]]
    non_filtered_attrs = [col for col in selected_attributes if not specific_values[col]]
    
    group_by_clause = ", ".join(filtered_attrs) if filtered_attrs else ""
    if group_by_clause:
        group_by_clause = f"GROUP BY {group_by_clause}"
    
    cube_clause = f"CUBE ({', '.join(non_filtered_attrs)})" if non_filtered_attrs else ""
    
    grouping_clause = f"{group_by_clause} {cube_clause}" if group_by_clause or cube_clause else ""
    
    any_value_clause = ", ".join([f"ANY_VALUE({col}) AS {col}" for col in selected_attributes])
    
    rollup_filter = []
    for col in filtered_attrs:
        rollup_filter.append(f"{col} IS NOT NULL")
    
    rollup_filter_clause = " AND ".join(rollup_filter)
    if rollup_filter_clause:
        rollup_filter_clause = f"WHERE {rollup_filter_clause}"
    
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
        visitors,
        purchases,
        total_revenue,
        conversion_rate,
        RANK() OVER (ORDER BY total_revenue DESC NULLS LAST) AS rank
    FROM (
        SELECT 
            {any_value_clause},
            COUNT(*) AS visitors,
            SUM(Purchase) AS purchases,
            SUM(Revenue) AS total_revenue,
            ROUND(SUM(Purchase) * 1.0 / COUNT(*), 2) AS conversion_rate
        FROM cleaned
        {grouping_clause}
        HAVING COUNT(*) >= {min_visitors}
    ) grouped
    {rollup_filter_clause}
    ORDER BY total_revenue DESC NULLS LAST
    """
    
    try:
        result = con.execute(query).fetchdf()
        if result.empty:
            st.warning(f"No groups meet the minimum visitor threshold of {min_visitors}.")
        else:
            st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Query error: {str(e)}")
