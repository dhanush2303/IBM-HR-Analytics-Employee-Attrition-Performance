import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset [cite: 14]
df = pd.read_csv('/Users/dhanushadurukatla/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Display the first few rows to ensure it's loaded correctly [cite: 14]
df.head()

# Check the shape (rows, columns) of the DataFrame [cite: 24]
print(f"Dataset shape: {df.shape}")

# Check for any duplicated rows [cite: 25]
print(f"Number of duplicated data: {df.duplicated().sum()}")

# Check for missing values in each column [cite: 26]
print("Missing values per column:")
print(df.isnull().sum())

# Check the data types of the columns [cite: 28]
print("\nColumn data types:")
df.dtypes

# Calculate the attrition rate [cite: 50]
attrition_rate = df['Attrition'].value_counts(normalize=True)
print("Attrition Rate:")
print(attrition_rate)

# Visualize the distribution of the attrition rate [cite: 51]
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=attrition_rate.index, y=attrition_rate.values)

# Add percentage labels to the bars [cite: 51]
for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')


plt.title('Distribution of Attrition Rate')
plt.xlabel('Attrition')
plt.ylabel('Percentage')
plt.tight_layout()
plt.show()

# Calculate the average years at the company [cite: 55]
avg_tenure = df['YearsAtCompany'].mean()

print(f'Average years of employee tenure is {avg_tenure:.2f} years')

# Create subplots for the three charts
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plot 1: Distribution of Employee by Age
sns.histplot(data=df, x='Age', kde=True, ax=axes[0])
axes[0].set_title('Distribution Employee by Age')

# Plot 2: Distribution of Employee by Gender
sns.countplot(data=df, x='Gender', ax=axes[1])
axes[1].set_title('Distribution Employee by Gender')

# Plot 3: Distribution of Employee by Department [cite: 59]
sns.countplot(data=df, x='Department', ax=axes[2])
axes[2].set_title('Distribution Employee by Department')

plt.tight_layout()
plt.show()

# Create a DataFrame containing only employees who have left [cite: 64]
df_attrition = df[df['Attrition'] == 'Yes']

# Set up the plots [cite: 34]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot 1: KDE plot of Age for employees who left [cite: 34]
sns.kdeplot(data=df_attrition, x='Age', fill=True, ax=axes[0])
axes[0].set_title('Attrition by Age')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Density')

# --- For Attrition Rate by Gender, we need a helper function ---
# This function calculates the attrition rate for any given column [cite: 71]
def calculate_attrition_rate(df, column):
    attrition_counts = df.groupby([column, 'Attrition']).size().unstack(fill_value=0)
    attrition_rate = (attrition_counts['Yes'] / attrition_counts.sum(axis=1)) * 100
    attrition_rate_df = attrition_rate.reset_index()
    attrition_rate_df.columns = [column, 'AttritionRate']
    return attrition_rate_df

# Calculate attrition rate by gender [cite: 34]
attrition_rate_gender = calculate_attrition_rate(df, 'Gender')

# Plot 2: Bar plot of Attrition Rate by Gender [cite: 34]
sns.barplot(data=attrition_rate_gender, x='Gender', y='AttritionRate', ax=axes[1])
axes[1].set_title('Attrition Rate by Gender')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Attrition Rate (%)')

plt.tight_layout()
plt.show()

# We will reuse the attrition rate calculation function from before
def calculate_attrition_rate(df, column):
    attrition_counts = df.groupby([column, 'Attrition']).size().unstack(fill_value=0)
    attrition_rate = (attrition_counts['Yes'] / attrition_counts.sum(axis=1)) * 100
    attrition_rate_df = attrition_rate.reset_index()
    attrition_rate_df.columns = [column, 'AttritionRate']
    return attrition_rate_df

# --- Create plots for each factor ---
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
fig.suptitle('Attrition Rate by Job Satisfaction and Work-Life Factors', fontsize=16)

# Plot 1: Attrition Rate by JobSatisfaction
satisfaction_rate = calculate_attrition_rate(df, 'JobSatisfaction')
# Map numeric values to labels for better readability as described in the PDF
satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
satisfaction_rate['JobSatisfaction'] = satisfaction_rate['JobSatisfaction'].map(satisfaction_map)
sns.barplot(data=satisfaction_rate, x='JobSatisfaction', y='AttritionRate', ax=axes[0], order=['Low', 'Medium', 'High', 'Very High'])
axes[0].set_title('by Job Satisfaction')
axes[0].set_ylabel('Attrition Rate (%)')

# Plot 2: Attrition Rate by JobInvolvement
involvement_rate = calculate_attrition_rate(df, 'JobInvolvement')
involvement_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
involvement_rate['JobInvolvement'] = involvement_rate['JobInvolvement'].map(involvement_map)
sns.barplot(data=involvement_rate, x='JobInvolvement', y='AttritionRate', ax=axes[1], order=['Low', 'Medium', 'High', 'Very High'])
axes[1].set_title('by Job Involvement')
axes[1].set_ylabel('') # Hide y-label for cleaner look

# Plot 3: Attrition Rate by WorkLifeBalance
worklife_rate = calculate_attrition_rate(df, 'WorkLifeBalance')
worklife_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
worklife_rate['WorkLifeBalance'] = worklife_rate['WorkLifeBalance'].map(worklife_map)
sns.barplot(data=worklife_rate, x='WorkLifeBalance', y='AttritionRate', ax=axes[2], order=['Bad', 'Good', 'Better', 'Best'])
axes[2].set_title('by Work-Life Balance')
axes[2].set_ylabel('') # Hide y-label

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Create plots for financial and role-based factors ---
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
fig.suptitle('Attrition by Financial and Role-Based Factors', fontsize=16)

# Plot 1: Distribution of Monthly Income by Attrition
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', ax=axes[0])
axes[0].set_title('Monthly Income Distribution by Attrition Status')
axes[0].set_xlabel('Attrition')
axes[0].set_ylabel('Monthly Income')

# Plot 2: Attrition Rate by Job Role
# We reuse the calculate_attrition_rate function from before
job_role_rate = calculate_attrition_rate(df, 'JobRole')
sns.barplot(data=job_role_rate, x='AttritionRate', y='JobRole', ax=axes[1], orient='h')
axes[1].set_title('Attrition Rate by Job Role')
axes[1].set_xlabel('Attrition Rate (%)')
axes[1].set_ylabel('Job Role')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# We will reuse the attrition rate calculation function
def calculate_attrition_rate(df, column):
    attrition_counts = df.groupby([column, 'Attrition']).size().unstack(fill_value=0)
    attrition_rate = (attrition_counts['Yes'] / attrition_counts.sum(axis=1)) * 100
    attrition_rate_df = attrition_rate.reset_index()
    attrition_rate_df.columns = [column, 'AttritionRate']
    return attrition_rate_df

# --- Create plots for each factor ---
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
fig.suptitle('Attrition Rate by Other Key Variables', fontsize=16)

# Plot 1: Attrition Rate by BusinessTravel
travel_rate = calculate_attrition_rate(df, 'BusinessTravel')
sns.barplot(data=travel_rate, x='BusinessTravel', y='AttritionRate', ax=axes[0])
axes[0].set_title('by Business Travel')
axes[0].set_ylabel('Attrition Rate (%)')

# Plot 2: Attrition Rate by DistanceFromHome (binned for clarity)
# We create bins to group distances together
bins = [0, 5, 10, 20, 30]
labels = ['0-5', '6-10', '11-20', '21+']
df['DistanceBin'] = pd.cut(df['DistanceFromHome'], bins=bins, labels=labels, right=False)
distance_rate = calculate_attrition_rate(df, 'DistanceBin')
sns.barplot(data=distance_rate, x='DistanceBin', y='AttritionRate', ax=axes[1])
axes[1].set_title('by Distance From Home (miles)')
axes[1].set_ylabel('')

# Plot 3: Attrition Rate by TotalWorkingYears (binned for clarity)
bins = [0, 5, 10, 20, 30, 45]
labels = ['0-5', '6-10', '11-20', '21-30', '31+']
df['ExperienceBin'] = pd.cut(df['TotalWorkingYears'], bins=bins, labels=labels, right=False)
experience_rate = calculate_attrition_rate(df, 'ExperienceBin')
sns.barplot(data=experience_rate, x='ExperienceBin', y='AttritionRate', ax=axes[2])
axes[2].set_title('by Total Working Years')
axes[2].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- 1. Prepare the Data ---

# Create a fresh copy of the original dataframe to avoid using the binned columns from the last step
df_model = df.drop(columns=['DistanceBin', 'ExperienceBin'])

# Convert the target variable 'Attrition' to numeric (Yes=1, No=0)
df_model['Attrition'] = df_model['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert all other object columns to numeric using Label Encoding
for column in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[column] = le.fit_transform(df_model[column])

# Define features (X) and target (y)
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

# --- 2. Split the Data ---

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Train a Logistic Regression Model ---
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
model.fit(X_train, y_train)

# --- 4. Evaluate the Model ---

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))

# Display the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- 1. Prepare the Data ---

# Create a fresh copy of the original dataframe to avoid using the binned columns from the last step
df_model = df.drop(columns=['DistanceBin', 'ExperienceBin'])

# Convert the target variable 'Attrition' to numeric (Yes=1, No=0)
df_model['Attrition'] = df_model['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert all other object columns to numeric using Label Encoding
for column in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[column] = le.fit_transform(df_model[column])

# Define features (X) and target (y)
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

# --- 2. Split the Data ---

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Train a Logistic Regression Model ---
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
model.fit(X_train, y_train)

# --- 4. Evaluate the Model ---

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))

# Display the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()