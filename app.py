import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Superkart Prediction App",
    page_icon="🛒",
    layout="wide"
)

# Load the trained model/pipeline
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
        # Build DataFrame with explicit data types
        input_data = pd.DataFrame({
            'Product_Weight': pd.Series([product_weight], dtype='float64'),
            'Product_Sugar_Content': pd.Series([str(product_sugar_content)], dtype='object'),
            'Product_Allocated_Area': pd.Series([product_allocated_area], dtype='float64'),
            'Product_Type': pd.Series([str(product_type)], dtype='object'),
            'Product_MRP': pd.Series([product_mrp], dtype='float64'),
            'Store_Establishment_Year': pd.Series([int(store_establishment_year)], dtype='int64'),
            'Store_Location_City_Type': pd.Series([str(store_location_city_type)], dtype='object'),
            'Store_Type': pd.Series([str(store_type)], dtype='object'),
            'Store_Age': pd.Series([int(store_age)], dtype='int64'),
            'Product_Category_Code': pd.Series([str(product_category_code)], dtype='object'),
            'Product_Type_Category': pd.Series([int(product_type_category)], dtype='int64'),
            'Product_Category_Prefix': pd.Series([str(product_category_prefix)], dtype='object'),
            'Store_Id': pd.Series([str(store_id)], dtype='object'),
            'Store_Size': pd.Series([str(store_size)], dtype='object')
        })

        try:
            # Make prediction
            prediction = model.predict(input_data)
            st.success(f"### Predicted Output: {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
