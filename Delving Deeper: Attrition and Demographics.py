import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- FIX: Load the dataset into a DataFrame called 'df' ---
# Make sure to replace 'path/to/your/...' with the correct file path on your computer.
df = pd.read_csv('/Users/dhanushadurukatla/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# --- Your Original Code (Now it will work) ---

# Attrition by Age
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Age', hue='Attrition', fill=True)
plt.title('Attrition by Age')
plt.show()

# Attrition Rate by Gender
attrition_by_gender = df.groupby('Gender')['Attrition'].value_counts(normalize=True).unstack()
print(f"\nAttrition Rate by Gender:\n{attrition_by_gender['Yes'] * 100}")

plt.figure(figsize=(8, 6))
sns.barplot(x=attrition_by_gender.index, y=attrition_by_gender['Yes'])
plt.title('Attrition Rate by Gender')
plt.ylabel('Attrition Rate')
plt.show()