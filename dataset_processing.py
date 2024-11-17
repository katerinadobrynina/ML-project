import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

loan_data = pd.read_csv('dataset/loan-10k.lrn.csv')
print(loan_data.head())
print(loan_data.isnull().sum())

sns.heatmap(loan_data.isnull(), cbar = False)
plt.title('Missing values')
plt.show()

print("desctiption")
print(loan_data.describe())
print(loan_data.info())
correlation_matrix = loan_data.corr()
target_corr = correlation_matrix['grade'].sort_values(ascending=False)
print(target_corr)
