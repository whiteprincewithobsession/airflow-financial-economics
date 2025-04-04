import pandas as pd

df = pd.read_csv('finance_economics_dataset.csv')
df.columns = [
    'Date', 'Stock_Index', 'Open_Price', 'Close_Price', 'Daily_High', 'Daily_Low', 
    'Trading_Volume', 'GDP_Growth_Pct', 'Inflation_Rate_Pct', 'Unemployment_Rate_Pct', 
    'Interest_Rate_Pct', 'Consumer_Confidence_Index', 'Government_Debt_B_USD', 
    'Corporate_Profits_B_USD', 'Forex_USD_EUR', 'Forex_USD_JPY', 'Crude_Oil_Price_USD', 
    'Gold_Price_USD', 'Real_Estate_Index', 'Retail_Sales_B_USD', 'Bankruptcy_Rate_Pct', 
    'Mergers_Acquisitions_Deals', 'Venture_Capital_Funding_B_USD', 'Consumer_Spending_B_USD'
]
df.to_csv('convenient_dataset.csv', index=False)