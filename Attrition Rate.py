import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- FIX: Load the dataset into a DataFrame called 'df' ---
# Make sure to replace 'path/to/your/...' with the correct file path on your computer.
df = pd.read_csv('/Users/dhanushadurukatla/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# --- Your Original Code (Now it will work) ---

# Calculate and visualize the attrition rate
attrition_rate = df['Attrition'].value_counts(normalize=True)
print(f"Attrition Rate:\n{attrition_rate * 100}")

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=attrition_rate.index, y=attrition_rate.values)
plt.title('Distribution of Attrition Rate')
plt.xlabel('Attrition')
plt.ylabel('Percentage')
for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.show()