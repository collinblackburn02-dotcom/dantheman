import duckdb
import streamlit as st
import pandas as pd
import os
from itertools import combinations

# Cache the combo analysis
@st.cache_data
def compute_combo_conversions(df, selected_attributes, specific_values, min_visitors=50):
    # Filter data based on specific values
    df_filtered = df.copy()
    for col, vals in specific_values.items():
        if vals and col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[column_mapping[col]].isin(vals)]
    
    # Generate all combinations (1 to 5 attributes)
    max_combo_size = 5
    combos = []
    for k in range(1, min(max_combo_size, len(selected_attributes)) + 1):
        combos.extend(list(combinations(selected_attributes, k)))
    
    # Collect results
    results = []
    for combo in combos:
        # Ensure all combo columns exist
        valid_combo = all(col in df_filtered.columns for col in combo)
        if not valid_combo:
            continue
        
        # Group by combo and calculate metrics
        mapped_combo = [column_mapping[col] for col in combo]
        group_df = df_filtered.groupby(mapped_combo).agg({
            'Purchase': ['sum', 'count'],
            'Revenue': 'sum'
        }).reset_index()
        group_df.columns = mapped_combo + ['purchases', 'visitors', 'total_revenue']
        # Exclude groups with blanks in combo attributes
        mask = group_df[mapped_combo].notna().all(axis=1) & (group_df[mapped_combo] != '').all(axis=1)
        group_df = group_df[mask]
        # Calculate conversion rate
        group_df['conversion_rate'] = group_df['purchases'] / group_df['visitors']
        group_df['combo_size'] = len(combo)
        # Filter by min visitors
        group_df = group_df[group_df['visitors'] >= min_visitors]
        # Rename back to UI attribute names
        group_df.columns = combo + ['purchases', 'visitors', 'total_revenue', 'conversion_rate', 'combo_size']
        if not group_df.empty:
            results.append(group_df)
    
    # Combine all results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        # Ensure non-selected attributes are blank
        for attr in ['Gender', 'Age_Range', 'Home_Owner', 'Net_Worth', 'Income_Range', 'State', 'Credit_Rating']:
            if attr not in selected_attributes:
                all_results[attr] = ''
        return all_results
    return pd.DataFrame()

# Debug: Print current directory and files
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Load your CSV (replace with your actual file path, e.g., 'data/Copy of DAN_HHS - Sample.csv')
try:
    df = pd.read_csv('Copy of DAN_HHS - Sample (1).csv')
    st.write("CSV loaded successfully with", len(df), "rows.")
    st.write("CSV columns:", df.columns.tolist())  # Debug: Show column names
except FileNotFoundError:
    st.error("CSV file 'Copy of DAN_HHS - Sample (1).csv' not found! Please check the file path or upload the file to the app directory.")
    st.stop()

# Streamlit UI
st.title("Ranked Customer Dashboard")

# Minimum visitors input
min_visitors = st.number_input(
    "Minimum number of visitors to show a group",
    min_value=0,
    value=50,
    step=1,
    help="Minimum visitors for analysis (cached). Your dataset has 197 rows."
)

# Available attributes (adjusted to match CSV columns)
available_attributes = ['Gender', 'Age_Range', 'Home_Owner', 'Net_Worth', 'Income_Range', 'State', 'Credit_Rating']

# Column mapping (to handle spaces in CSV columns)
column_mapping = {
    'Gender': 'Gender',
    'Age_Range': 'Age Range',
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
            include = st.checkbox(f"Include {attr}", value=True, key=f"checkbox_{attr}")
            if include:
                selected_attributes.append(attr)
                col_name = column_mapping[attr]
                unique_values = df[col_name].dropna().unique().tolist()
                specific_values[attr] = st.multiselect(
                    f"Values for {attr} (all if empty)",
                    options=unique_values,
                    default=[],
                    key=f"multiselect_{attr}"
                )

# Compute and cache combo conversions
combo_data = compute_combo_conversions(df, selected_attributes, specific_values, min_visitors)

# Display results
if not selected_attributes:
    st.warning("Please include at least one attribute to display the table.")
elif combo_data.empty:
    st.warning(f"No groups meet the minimum visitor threshold of {min_visitors}. Try a lower value (dataset has {len(df)} rows).")
else:
    # Apply UI filter on cached data (optional override)
    display_data = combo_data.copy()
    for col, vals in specific_values.items():
        if vals:
            display_data = display_data[display_data[col].isin(vals)]
    # Sort and rank
    display_data = display_data.sort_values(by=['conversion_rate', 'purchases', 'visitors'], ascending=[False, False, False])
    display_data['rank'] = range(1, len(display_data) + 1)
    st.dataframe(display_data)
