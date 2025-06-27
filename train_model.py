# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

print("Starting model training...")

# --- 1. Load and Prepare Data ---
# Make sure to replace 'path/to/your/...' with your actual file path
df = pd.read_csv('/Users/dhanushadurukatla/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df_model = df.copy()
df_model['Attrition'] = df_model['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert all other object columns to numeric using Label Encoding
for column in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[column] = le.fit_transform(df_model[column])

# Define features (X) and target (y)
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

# --- 2. Train the Model ---
# We train on the full dataset here to make the saved model as accurate as possible
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model training complete.")

# --- 3. Save the Trained Model to a File ---
joblib.dump(model, 'attrition_model.joblib')

print("Model has been saved successfully as attrition_model.joblib")