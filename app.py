import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Superkart Prediction App",
    page_icon="🛒",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("superkart_random_forest_model_v1_0.joblib")

model = load_model()

st.title("🛒 Superkart Sales Prediction App (v2)")
st.write("Enter the product and store details below to predict performance.")

# Create input form for single prediction
with st.form("prediction_form"):
    st.subheader("Single Product & Store Prediction Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.50, step=0.1)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Fat", "Regular", "Non-Edible"])
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        product_type = st.selectbox("Product Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods"])
        product_mrp = st.number_input("Product MRP", min_value=0.0, max_value=300.0, value=150.00, step=1.0)
        store_establishment_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2026, value=2005, step=1)
        store_id = st.selectbox("Store ID", ["OUT049", "OUT018", "OUT027", "OUT013", "OUT046", "OUT035", "OUT019", "OUT045", "OUT017", "OUT010"])

    with col2:
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
        store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
        store_age = st.number_input("Store Age (Years)", min_value=0, max_value=50, value=15, step=1)
        product_category_code = st.selectbox("Product Category Code", ["FD", "NC", "DR"])
        product_type_category = st.selectbox("Product Type Category", [1, 2, 3])
        product_category_prefix = st.selectbox("Product Category Prefix", ["FD", "NC", "DR"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Construct DataFrame ensuring categorical features are explicit strings and numeric are floats/ints
        input_data = pd.DataFrame({
            'Product_Weight': [float(product_weight)],
            'Product_Sugar_Content': [str(product_sugar_content)],
            'Product_Allocated_Area': [float(product_allocated_area)],
            'Product_Type': [str(product_type)],
            'Product_MRP': [float(product_mrp)],
            'Store_Establishment_Year': [int(store_establishment_year)],
            'Store_Location_City_Type': [str(store_location_city_type)],
            'Store_Type': [str(store_type)],
            'Store_Age': [int(store_age)],
            'Product_Category_Code': [str(product_category_code)],
            'Product_Type_Category': [int(product_type_category)],
            'Product_Category_Prefix': [str(product_category_prefix)],
            'Store_Id': [str(store_id)],
            'Store_Size': [str(store_size)]
        })

        try:
            # Make prediction
            prediction = model.predict(input_data)
            st.success(f"### Predicted Output: {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
