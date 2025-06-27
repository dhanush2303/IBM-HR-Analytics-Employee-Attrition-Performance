# (This code should be run after you have trained the model in Step 8)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Re-run model training steps to ensure we have the model object ---
df_model = df.copy().drop(columns=['DistanceBin', 'ExperienceBin'], errors='ignore')
df_model['Attrition'] = df_model['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
for column in df_model.select_dtypes(include='object').columns:
    df_model[column] = LabelEncoder().fit_transform(df_model[column])
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Save the trained model to a file ---
joblib.dump(model, 'attrition_model.joblib')

print("Model saved successfully as attrition_model.joblib")