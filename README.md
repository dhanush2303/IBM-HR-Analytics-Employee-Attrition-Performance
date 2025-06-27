IBM HR Analytics Employee Attrition & Performance
Project Goal: The primary objective was to analyze a dataset of fictional employee data to understand the key factors driving employee attrition and to build a predictive model to identify employees at risk of leaving.

Step-by-Step Implementation:

Step 1: Data Cleaning and Initial Exploration

We loaded the dataset and performed initial checks for shape, duplicates, and missing values.

We confirmed the dataset was clean and ready for analysis.

Step 2: Foundational Analysis

We calculated and visualized the overall company attrition rate, finding it to be 16.12%.

We analyzed the company's demographic landscape, looking at the distribution of age, gender, and department.

Step 3: Deep-Dive Factor Analysis

To identify the key drivers of attrition, we analyzed several factors, including:

Job Satisfaction: We found that employees with lower job satisfaction, involvement, and poor work-life balance had significantly higher attrition rates.

Financial Factors: We determined that lower monthly income and specific job roles (like Sales Representative) were strongly linked to leaving.

Other Key Variables: We discovered that frequent business travel and having fewer total working years also contributed to higher attrition.

Step 4: Predictive Modeling (Machine Learning)

We built a Logistic Regression model to predict the likelihood of an employee leaving.

This involved preparing the data, splitting it into training and testing sets, and evaluating the model's performance with a classification report and confusion matrix.

Step 5: Deployment (Web Application)

As a final, advanced step, we provided the complete code and step-by-step instructions to build a simple, interactive web application using Streamlit.

We debugged the execution process, clarifying that the app needed to be run from the terminal, not within a notebook. This app allows a user to input an employee's details and get an instant attrition risk prediction.

