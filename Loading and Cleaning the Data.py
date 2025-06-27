import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('/Users/dhanushadurukatla/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# --- Initial Data Exploration ---

# Check the shape of the data
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# Check for duplicated data
print(f"Number of duplicated rows: {df.duplicated().sum()}")

# Check for missing values
print("Percentage of missing values in each column:")
print(df.isnull().sum() / len(df) * 100)

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Get a statistical summary of the numerical columns
print("\nStatistical summary of the dataset:")
print(df.describe())