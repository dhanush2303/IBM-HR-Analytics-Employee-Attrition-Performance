IBM HR Employee Attrition Analysis & Prediction:

Project Goal: The primary objective was to analyze a dataset of employee information to identify the key factors that lead to employee attrition (employees leaving the company). The second goal was to build a predictive machine learning model that could forecast the likelihood of any given employee leaving, creating a valuable tool for the HR department.

Step-by-Step Implementation Summary
Step 1: Data Loading and Initial Checks

We began by loading the dataset using pandas.

We performed initial checks for data shape, duplicates, and missing values, confirming the dataset was clean and ready for analysis.




Step 2: Exploratory Data Analysis (EDA) to Find Key Factors

We calculated the overall company attrition rate, finding that 16.12% of employees had left.

We performed a deep-dive analysis to understand why employees were leaving, focusing on:

Job Satisfaction: We found a strong link between lower job satisfaction, lower job involvement, poor work-life balance, and a higher rate of attrition.

Financial Factors: We discovered that employees with lower monthly incomes were more likely to leave. We also identified specific job roles, like Sales Representative, that had dramatically higher attrition rates.

Other Variables: We analyzed other factors and found that frequent business travel and fewer years of total work experience also correlated with a higher chance of leaving.

Step 3: Building the Predictive Model

We created a machine learning model using LogisticRegression to predict employee attrition.

This involved encoding the categorical data (like gender, job role, etc.) into a numerical format that the model could understand.

We evaluated the model's performance to ensure it was accurate and effective.

Step 4: Saving the Model for the Application

We created a separate Python script (train_model.py) for the sole purpose of training the model on the full dataset and saving the final, trained model into a single file named attrition_model.joblib. This step is crucial for the app to work.

The Final Outcome: The Web Application
The final result of our project is an interactive web application that serves as a practical tool for an HR manager. The app allows a user to:

Enter the details of a current employee using sliders and dropdown menus.

Click a "Predict Attrition" button.

Instantly see a prediction from our trained model, indicating whether the employee is at a "High Risk" or "Low Risk" of leaving, along with a probability score.

How to Run the Application
To run the app on your computer, you must follow this two-step process in your terminal from your project directory.

Step 1: Create the Model File (Run this only once)
Before you can run the app, you must first create the attrition_model.joblib file. If you haven't already, run the training script from your terminal:

Bash

python train_model.py
This will create the necessary model file in your project folder.

Step 2: Launch the Streamlit App
Once the attrition_model.joblib file exists, you can start the web application. Run the following command in your terminal:

Bash

streamlit run app.py
After running this command, a new tab will automatically open in your web browser at an address like http://localhost:8501. You will see your live, interactive application there.
