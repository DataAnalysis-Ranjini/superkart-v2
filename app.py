import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.title("SuperKart Sales Predictor")

# Load your trained model (update the filename if yours is named differently)
@st.cache_resource
def load_model():
    return joblib.load('superkart_random_forest_model_v1_0.joblib')

model = load_model()

# 1. Gather inputs from your Streamlit form widgets
prod_weight = st.number_input("Product Weight", value=10.0)
allocated_area = st.number_input("Product Allocated Area", value=0.1)
mrp = st.number_input("Product MRP", value=150.0)
establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2026, value=2010, step=1)

# Calculate Store_Age dynamically just like in training
current_year = datetime.now().year
store_age = current_year - int(establishment_year)

# Categorical inputs using selectboxes and text inputs
sugar_content = st.selectbox("Product Sugar Content", ["Low Fat", "Regular", "Non-Edible"])
product_type = st.selectbox("Product Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods"])
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
category_code = st.text_input("Product Category Code", value="DR")
store_id = st.text_input("Store ID", value="OUT049")
product_id = st.text_input("Product ID", value="FDA15")
product_type_category = st.text_input("Product Type Category", value="1")

# Automatically derive Product_Category_Prefix from the first 2 letters of Product_Id
product_category_prefix = str(product_id)[:2].upper()

# 2. Build the input DataFrame with ALL required features including Product_Category_Prefix
input_data = {
    'Product_Weight': float(prod_weight),
    'Product_Allocated_Area': float(allocated_area),
    'Product_MRP': float(mrp),
    'Store_Age': int(store_age),
    'Product_Sugar_Content': str(sugar_content),
    'Product_Type': str(product_type),
    'Store_Size': str(store_size),
    'Store_Location_City_Type': str(city_type),
    'Store_Type': str(store_type),
    'Product_Category_Code': str(category_code),
    'Store_Id': str(store_id),
    'Product_Id': str(product_id),
    'Product_Type_Category': str(product_type_category),
    'Product_Category_Prefix': str(product_category_prefix)
}

input_df = pd.DataFrame([input_data])

# 3. Predict button
if st.button("Predict Sales"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Sales Total: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
