import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from lightgbm import LGBMClassifier

# Load the LightGBM model using pickle
model_path = 'models/lgbmodel.pkl'

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Function to preprocess data and predict stroke probability
    def predict_stroke_probability(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                                    avg_glucose_level, bmi, smoking_status):
        # Convert categorical variables to numeric using one-hot encoding
        work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        smoking_statuses = ['formerly smoked', 'never smoked', 'smokes']

        work_type_encoded = [1 if work_type == wt else 0 for wt in work_types]
        smoking_status_encoded = [1 if smoking_status == ss else 0 for ss in smoking_statuses]

        # Create input DataFrame for prediction
        data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'work_type_Private': [work_type_encoded[0]],
            'work_type_Self-employed': [work_type_encoded[1]],
            'work_type_Govt_job': [work_type_encoded[2]],
            'work_type_children': [work_type_encoded[3]],
            'work_type_Never_worked': [work_type_encoded[4]],
            'smoking_status_formerly smoked': [smoking_status_encoded[0]],
            'smoking_status_never smoked': [smoking_status_encoded[1]],
            'smoking_status_smokes': [smoking_status_encoded[2]],
        })

        # Select only the features used during model training
        features_used = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
                         'avg_glucose_level', 'bmi', 'work_type_Private', 'work_type_Self-employed', 'work_type_Govt_job',
                         'work_type_children', 'work_type_Never_worked', 'smoking_status_formerly smoked',
                         'smoking_status_never smoked', 'smoking_status_smokes']

        data = data[features_used]

        # Make prediction
        probability = model.predict_proba(data)[:, 1]  # Probability of stroke (class 1)
        
        # Multiply probability by 100 to get percentage
        probability *= 100

        return probability[0]

    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
    if 'probability' not in st.session_state:
        st.session_state.probability = None

    def main():
        # Page: Input
        if st.session_state.page == 'input':
            st.title('Stroke Probability Prediction')
            st.write('This app predicts the probability of stroke based on input features.')

            # Input fields for user input
            gender = st.selectbox('Gender', ['Male', 'Female'])
            age = st.slider('Age', 0, 150, 50)
            hypertension = st.checkbox('Hypertension')
            heart_disease = st.checkbox('Heart Disease')
            ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
            work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
            residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
            avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=500.0, value=100.0, step=0.1)
            bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])

            # Convert input variables to numeric
            residence_num = 1 if residence_type == 'Urban' else 0
            gender_num = 1 if gender == 'Male' else 0
            ever_married_num = 1 if ever_married == 'Yes' else 0
            hypertension_num = 1 if hypertension else 0
            heart_disease_num = 1 if heart_disease else 0

            # Predict stroke probability when 'Predict' button is clicked
            if st.button('Predict'):
                probability = predict_stroke_probability(gender_num, age, hypertension_num, heart_disease_num, ever_married_num,
                                                         work_type, residence_num, avg_glucose_level, bmi, smoking_status)

                st.session_state.probability = probability
                st.session_state.page = 'result'
                st.experimental_rerun()

        # Page: Result
        elif st.session_state.page == 'result':
            probability = st.session_state.probability
            st.write(f'Probability of stroke: {probability:.2f}%')  # Display probability as percentage

            st.write("### Was this result correct?")
            if st.button('Yes'):
                st.session_state.page = 'feedback'
                st.experimental_rerun()
            if st.button('No'):
                st.session_state.page = 'feedback'
                st.experimental_rerun()

        # Page: Feedback
        elif st.session_state.page == 'feedback':
            st.write("Thank you for your feedback!")
            if st.button('Go back to input'):
                st.session_state.page = 'input'
                st.experimental_rerun()

    # Run the app
    if __name__ == '__main__':
        main()
