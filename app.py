# ... (inside your app.py Toggle Area) ...

col1, col2, col3, col4 = st.columns(4)
with col1: 
    inc_gender = st.checkbox("Gender", value=True)
    inc_children = st.checkbox("Children", value=False)
with col2: 
    inc_age = st.checkbox("Age Range", value=True)
    inc_nw = st.checkbox("Net Worth", value=False)
with col3: 
    inc_income = st.checkbox("Income Range", value=True)
with col4: 
    inc_state = st.checkbox("State", value=False)

# Build the filter list based on checked boxes
active_types = []
if inc_gender: active_types.append('gender')
if inc_age: active_types.append('age')
if inc_income: active_types.append('income')
if inc_state: active_types.append('state')
if inc_nw: active_types.append('net_worth')
if inc_children: active_types.append('children')

# Add pre-calculated combos
if inc_gender and inc_age: active_types.append('gender_age')
if inc_gender and inc_income: active_types.append('gender_income')
if inc_gender and inc_nw: active_types.append('gender_nw')
if inc_gender and inc_children: active_types.append('gender_children')
if inc_gender and inc_age and inc_income: active_types.append('gender_age_income')

# ... (rest of the filtering and display code) ...
