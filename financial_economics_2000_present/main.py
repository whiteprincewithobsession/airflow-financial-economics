import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('convenient_dataset.csv')

df['Date'] = pd.to_datetime(df['Date'])

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    median_val = df[column].median()
    df[column] = df[column].fillna(median_val)

df = df.drop_duplicates()

df['Daily_Volatility'] = df['Daily_High'] - df['Daily_Low']
df['Daily_Return_Pct'] = (df['Close_Price'] - df['Open_Price']) / df['Open_Price'] * 100
df['7D_MA_Close'] = df['Close_Price'].rolling(window=7).mean()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

df.to_csv('processed_financial_data.csv', index=False)