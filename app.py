import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import tensorflow as tf

## Load the trained model
model = tf.keras.models.load_model('model.h5')

## Load the label encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## streamlit app
st.title("Customer Churn Prediction")

# user format
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
tenure = st.slider("Tenure", 0, 10)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
credit_score = st.slider("Credit Score")
estimated_salary = st.number_input("Estimated Salary")

## prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

input_scaled = scaler.transform(input_data)


prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {1 - prediction_prob:.2f}")