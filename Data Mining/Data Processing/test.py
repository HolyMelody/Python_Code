import pandas as pd


df = pd.read_csv('data_features_02.csv')
print(type(df))
print(df.iloc[1,1])