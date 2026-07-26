import pandas as pd
from datetime import datetime

# 1. Gather inputs from your Streamlit form
prod_weight = float(...)
allocated_area = float(...)
mrp = float(...)
establishment_year = int(...) # or from a selectbox/number_input

# Calculate Store_Age dynamically just like in training
current_year = datetime.now().year
store_age = current_year - establishment_year

# 2. Build the input DataFrame with ALL required features in the correct names
input_data = {
    'Product_Weight': prod_weight,
    'Product_Allocated_Area': allocated_area,
    'Product_MRP': mrp,
    'Store_Age': store_age,
    'Product_Sugar_Content': str(...),
    'Product_Type': str(...),
    'Store_Size': str(...),
    'Store_Location_City_Type': str(...),
    'Store_Type': str(...),
    'Product_Category_Code': str(...),
    'Store_Id': str(...),
    'Product_Id': str(...),
    'Product_Type_Category': str(...) # or numeric depending on how you encoded it
}

input_df = pd.DataFrame([input_data])

# 3. Predict
prediction = model.predict(input_df)
