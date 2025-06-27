import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- FIX: Load the dataset into a DataFrame called 'df' ---
# Make sure to replace 'path/to/your/...' with the correct file path on your computer.
df = pd.read_csv('/Users/dhanushadurukatla/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# --- Your Original Code (Now it will work) ---

# Visualize employee demographics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age distribution
sns.histplot(df['Age'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Employee Age')

# Gender distribution
sns.countplot(x='Gender', data=df, ax=axes[1])
axes[1].set_title('Distribution of Employee by Gender')

# Department distribution
sns.countplot(x='Department', data=df, ax=axes[2])
axes[2].set_title('Distribution of Employee by Department')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()