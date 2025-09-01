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
        if vals:
            df_filtered = df_filtered[df_filtered[column_mapping[col]].isin(vals)]
    
    # Generate all combinations (1 to 5 attributes)
    max_combo_size = 5
    combos = []
    for k in range(1, min(max_combo_size, len(selected_attributes)) + 1):
        combos.extend(list(combinations(selected_attributes, k)))
    
    # Collect results
    results = []
    for combo in combos:
        mapped_combo = [column_mapping[col] for col in combo]
        # Group by combo and calculate metrics
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
        for attr in available_attributes:
            if attr not in selected_attributes:
                all_results[attr] = ''
        return all_results
    return pd.DataFrame()

# Debug: Print current directory and files
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Load your CSV (replace with your actual file path, e.g., 'data/Copy of DAN_HHS - Sample.csv')
try:
    df = pd.read_csv('Copy of DAN_HHS - Sample.csv')
    st.write("CSV loaded successfully with", len(df), "rows.")
    st.write("CSV columns:", df.columns.tolist())  # Debug: Show column names
except FileNotFoundError:
    st.error("CSV file 'Copy of DAN_HHS - Sample.csv' not found! Please check the file path or upload the file to the app directory.")
    st.stop()

# Streamlit UI
st.title("Ranked Customer Dashboard")

# Minimum visitors input
min_visitors = st.number_input(
    "Minimum number of visitors to show a group",
    min_value=0,
    value=50,
    step=1,
    help="
