# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('attrition_model.joblib')

# --- Create the UI of the App ---

st.title('Employee Attrition Prediction App')
st.write('This app predicts the likelihood of an employee leaving the company.')
st.write('Please enter the employee\'s details below to get a prediction.')

# Create input fields for the features used by the model
# We will create dropdowns for categorical features and number inputs for numerical ones.
# The options for dropdowns should match the data the model was trained on.

st.header('Employee Details')

# Categorical Inputs
business_travel = st.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox('Department', ['Human Resources', 'Research & Development', 'Sales'])
education_field = st.selectbox('Education Field', ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other',
                                                   'Technical Degree'])
gender = st.selectbox('Gender', ['Female', 'Male'])
job_role = st.selectbox('Job Role', ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
                                     'Manufacturing Director', 'Research Director', 'Research Scientist',
                                     'Sales Executive', 'Sales Representative'])
marital_status = st.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
over_time = st.selectbox('Over Time', ['No', 'Yes'])

# Numerical Inputs
age = st.slider('Age', 18, 60, 35)
daily_rate = st.slider('Daily Rate', 100, 1500, 800)
distance_from_home = st.slider('Distance From Home (miles)', 1, 30, 10)
job_satisfaction = st.slider('Job Satisfaction (1-4)', 1, 4, 3)
monthly_income = st.slider('Monthly Income ($)', 1000, 20000, 6500)
total_working_years = st.slider('Total Working Years', 0, 40, 10)
years_at_company = st.slider('Years at Company', 0, 40, 7)
years_with_curr_manager = st.slider('Years with Current Manager', 0, 20, 4)

# --- Prediction Logic ---
if st.button('Predict Attrition'):
    # Prepare the input data for the model
    # The model expects a DataFrame with the same columns and encoding as the training data

    # Create a dictionary for the LabelEncoders (this is a simplified approach)
    # In a real-world scenario, you would save and load the fitted encoders
    encoding_maps = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 2, 'Travel_Frequently': 1},
        'Department': {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
        'EducationField': {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2, 'Medical': 3, 'Other': 4,
                           'Technical Degree': 5},
        'Gender': {'Female': 0, 'Male': 1},
        'JobRole': {'Healthcare Representative': 0, 'Human Resources': 1, 'Laboratory Technician': 2, 'Manager': 3,
                    'Manufacturing Director': 4, 'Research Director': 5, 'Research Scientist': 6, 'Sales Executive': 7,
                    'Sales Representative': 8},
        'MaritalStatus': {'Divorced': 0, 'Married': 1, 'Single': 2},
        'OverTime': {'No': 0, 'Yes': 1}
    }

    input_data = {
        'Age': age, 'BusinessTravel': encoding_maps['BusinessTravel'][business_travel],
        'DailyRate': daily_rate, 'Department': encoding_maps['Department'][department],
        'DistanceFromHome': distance_from_home, 'Education': 1,  # Placeholder
        'EducationField': encoding_maps['EducationField'][education_field], 'EmployeeCount': 1,  # Placeholder
        'EmployeeNumber': 1, 'EnvironmentSatisfaction': 1, 'Gender': encoding_maps['Gender'][gender],  # Placeholder
        'HourlyRate': 1, 'JobInvolvement': 1, 'JobLevel': 1,  # Placeholder
        'JobRole': encoding_maps['JobRole'][job_role], 'JobSatisfaction': job_satisfaction,
        'MaritalStatus': encoding_maps['MaritalStatus'][marital_status], 'MonthlyIncome': monthly_income,
        'MonthlyRate': 1, 'NumCompaniesWorked': 1, 'Over18': 1, 'OverTime': encoding_maps['OverTime'][over_time],
        # Placeholder
        'PercentSalaryHike': 1, 'PerformanceRating': 1, 'RelationshipSatisfaction': 1,  # Placeholder
        'StandardHours': 80, 'StockOptionLevel': 1, 'TotalWorkingYears': total_working_years,  # Placeholder
        'TrainingTimesLastYear': 1, 'WorkLifeBalance': 1, 'YearsAtCompany': years_at_company,  # Placeholder
        'YearsInCurrentRole': 1, 'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': years_with_curr_manager
        # Placeholder
    }

    # Create a DataFrame from the input
    input_df = pd.DataFrame([input_data])
    # The model was trained on 34 columns, so ensure this DataFrame matches that structure
    # NOTE: We've simplified and used placeholders for some features for this example app.
    # To get perfect accuracy, all input fields should be present.

    # Reorder columns to match the training data
    expected_columns = model.feature_names_in_
    input_df = input_df[expected_columns]

    # Get prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader('Prediction Result')
    if prediction == 1:
        st.error(f'High Risk of Attrition (Probability: {prediction_proba[1] * 100:.2f}%)')
        st.write("The model predicts that this employee is likely to leave.")
    else:
        st.success(f'Low Risk of Attrition (Probability: {prediction_proba[0] * 100:.2f}%)')
        st.write("The model predicts that this employee is likely to stay.")