import duckdb
import streamlit as st
import pandas as pd
import os
from itertools import combinations

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
    help="Your dataset has 197 rows, so groups can't exceed 197 visitors. Set to 400 for larger datasets."
)

# Available attributes (based on your scripts)
available_attributes = ['Gender', 'Age_Range', 'Income_Range', 'Net_Worth', 'Home_Owner', 'Married', 'Children', 'Credit_Rating', 'State']

# Map display names to CSV column names
column_mapping = {
    'Gender': 'Gender',
    'Age_Range': 'Age Range',
    'Income_Range': 'Income Range',
    'Net_Worth': 'New Worth',
    'Home_Owner': 'Home Owner',
    'Married': 'Married',
    'Children': 'Children',
    'Credit_Rating': 'Credit Rating',
    'State': 'State'
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

# Build results
if not selected_attributes:
    st.warning("Please include at least one attribute to display the table.")
else:
    # Construct WHERE clause for specific values (initial filter)
    where_clauses = []
    for col, vals in specific_values.items():
        if vals:
            vals_str = ", ".join([f"'{v}'" for v in vals])
            where_clauses.append(f'"{column_mapping[col]}" IN ({vals_str})')

    where_clause = " AND ".join(where_clauses)
    if where_clause:
        where_clause = f"WHERE {where_clause}"
    
    # Generate combinations (1 to 5 attributes, like your scripts)
    max_combo_size = 5
    combos = []
    for k in range(1, min(max_combo_size, len(selected_attributes)) + 1):
        combos.extend(list(combinations(selected_attributes, k)))

    # Collect results
    results = []
    for combo in combos:
        # Construct SELECT and GROUP BY for this combo
        select_clause = ", ".join([f'"{column_mapping[col]}" AS {col}' for col in combo])
        group_by_clause = ", ".join(combo)
        query = f"""
        SELECT 
            {select_clause},
            COUNT(*) AS visitors,
            SUM(Purchase) AS purchases,
            SUM(Revenue) AS total_revenue,
            ROUND(SUM(Purchase) * 1.0 / COUNT(*), 2) AS conversion_rate
        FROM customers
        {where_clause}
        GROUP BY {group_by_clause}
        HAVING COUNT(*) >= {min_visitors}
        """
        try:
            result = con.execute(query).fetchdf()
            if not result.empty:
                result['combo_size'] = len(combo)
                results.append(result)
        except Exception as e:
            st.error(f"Query error for combo {combo}: {e}")

    # Combine all results and rank
    if results:
        all_results = pd.concat(results, ignore_index=True)
        all_results = all_results.sort_values(by=['conversion_rate', 'purchases', 'visitors'], ascending=False)
        all_results['rank'] = range(1, len(all_results) + 1)
        st.dataframe(all_results)
    else:
        st.warning(f"No groups meet the minimum visitor threshold of {min_visitors}. Try a lower value (dataset has {len(df)} rows).")
