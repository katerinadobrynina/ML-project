import pandas as pd
import matplotlib.pyplot as plt

file_pth = "AmesHousing.csv"
df = pd.read_csv(file_pth)



def basic_data_analysis(df):
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of attributes: {df.shape[1] - 2}")  # -2 because of order and PID attrs

def search_missing_values(df):
    missing_values = df.isnull().sum().sum()
    print(f"Missing Values: {missing_values}")
    df.info(), df.head()


nominal_columns = df.select_dtypes(include=['object', 'category']).columns
numeric_columns = df.select_dtypes(include=['number']).columns

print("Nominal columns:", nominal_columns)
print("Numeric columns:", numeric_columns)


categ_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols].hist(bins=50, figsize=(50, 50), color = ["red"], edgecolor='black')
plt.savefig("Histograma.png")


fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(categ_cols[:9]):  # Limiting to 9 categorical attributes for visualization
    df[col].value_counts().plot(kind='bar', ax=axes[i], color='green', edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.savefig("CATEGORIES.png")

min_value = df['SalePrice'].min()
max_value = df['SalePrice'].max()
sale_price_summary = df['SalePrice'].describe()
print(f"Median price: {sale_price_summary.median()}\n"
      f"Mean: {sale_price_summary.mean().round()}\n"
      f"Standard deviation: {sale_price_summary.std().round()}\n"
      f"Min value: {min_value}\n"
      f"Max value: {max_value}")

plt.figure(figsize=(10, 6))
df['SalePrice'].hist(bins=20, color='blue', edgecolor='black')
plt.title('Distribution of SalePrice', fontsize=16)
plt.xlabel('SalePrice', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(False)
plt.savefig("SalePrice.png")



basic_data_analysis(df)
search_missing_values(df)