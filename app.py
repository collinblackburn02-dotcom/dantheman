# ================ Inside app.py: Dynamic Type Resolver =================
# This logic automatically picks the right 'cluster_type' based on what is 'Inc'
inc_list = sorted(included_types)  # included_types comes from the checkboxes
active_types = []

# 1. Add Single Type
if len(inc_list) >= 1:
    active_types.extend(inc_list)

# 2. Add Two-Way Logic (All possible pairs)
if len(inc_list) >= 2:
    for i in range(len(inc_list)):
        for j in range(i + 1, len(inc_list)):
            # Special naming fix for net_worth vs nw
            t1 = inc_list[i].replace("net_worth", "nw")
            t2 = inc_list[j].replace("net_worth", "nw")
            active_types.append(f"{inc_list[i]}_{inc_list[j]}")
            active_types.append(f"{inc_list[i]}_{t2}") # Catch nw aliases

# 3. Add Three-Way Logic (Matches the unions we added in SQL)
# We add them manually to ensure they match the SQL strings exactly
if "gender" in inc_list and "age" in inc_list and "income" in inc_list: active_types.append("gender_age_income")
if "gender" in inc_list and "age" in inc_list and "state" in inc_list: active_types.append("gender_age_state")
if "gender" in inc_list and "income" in inc_list and "state" in inc_list: active_types.append("gender_income_state")
if "gender" in inc_list and "income" in inc_list and "net_worth" in inc_list: active_types.append("gender_income_nw")
if "gender" in inc_list and "age" in inc_list and "children" in inc_list: active_types.append("gender_age_children")
if "age" in inc_list and "income" in inc_list and "net_worth" in inc_list: active_types.append("age_income_nw")
if "state" in inc_list and "income" in inc_list and "net_worth" in inc_list: active_types.append("state_income_nw")
if "gender" in inc_list and "state" in inc_list and "net_worth" in inc_list: active_types.append("gender_state_nw")
