import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier

# Load the LightGBM model
model = joblib.load('models/lgbmodel.pkl')

# Function to preprocess data and predict stroke probability
def predict_stroke_probability(gender, age, hypertension, heart_disease, work_type, residence_type,
                                avg_glucose_level, bmi, smoking_status):
    # Convert categorical variables to numeric using one-hot encoding
    work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    smoking_statuses = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

    work_type_encoded = [1 if work_type == wt else 0 for wt in work_types]
    smoking_status_encoded = [1 if smoking_status == ss else 0 for ss in smoking_statuses]

    # Create input DataFrame for prediction
    data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'work_type_Private': [work_type_encoded[0]],
        'work_type_Self-employed': [work_type_encoded[1]],
        'work_type_Govt_job': [work_type_encoded[2]],
        'work_type_children': [work_type_encoded[3]],
        'smoking_status_formerly smoked': [smoking_status_encoded[0]],
        'smoking_status_never smoked': [smoking_status_encoded[1]],
        'smoking_status_smokes': [smoking_status_encoded[2]],
    })

    # Select only the features used during model training
    features_used = ['gender', 'age', 'hypertension', 'heart_disease', 'Residence_type',
                     'avg_glucose_level', 'bmi', 'work_type_Private', 'smoking_status_formerly smoked']

    # Filter input data to include only the required features
    data = data[features_used]

    # Make prediction
    probability = model.predict_proba(data)[:, 1]  # Probability of stroke (class 1)
    
    # Multiply probability by 100
    probability *= 100

    return probability[0]

# Streamlit app begins
def main():
    st.title('Stroke Probability Prediction')
    st.write('This app predicts the probability of stroke based on input features.')

    # Input fields for user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 0, 150, 50)
    hypertension = st.checkbox('Hypertension')
    heart_disease = st.checkbox('Heart Disease')
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children'])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=500.0, value=100.0, step=0.1)
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])

    residence_num = 1 if residence_type == 'Urban' else 0
    gender_num = 1 if gender == 'Male' else 0

    # Predict stroke probability when 'Predict' button is clicked
    if st.button('Predict'):
        probability = predict_stroke_probability(gender_num, age, hypertension, heart_disease, work_type, residence_num,
                                                 avg_glucose_level, bmi, smoking_status)

        st.write(f'Probability of stroke: {probability:.2f}%')  # Display probability as percentage

        # Provide advice based on probability
        if probability > 70:
            st.write("Based on the prediction, it is advised to seek medical attention.")
        elif probability > 50:
            st.write("Based on the prediction, it is advised to have a check-up.")
        elif probability >= 40:
            st.write("Based on the prediction, it is probably fine.")
        else:
            st.write("Based on the prediction, there are likely no problems.")

# Run the app
if __name__ == '__main__':
    main()
