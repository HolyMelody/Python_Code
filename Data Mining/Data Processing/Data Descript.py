



import pandas as pd

df = pd.read_csv('data.csv')
Wave_Height__Period=df.iloc[:,2:7]
Q1 = Wave_Height__Period.quantile(0.25)
Q3 = Wave_Height__Period.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = Wave_Height__Period[~((Wave_Height__Period < (Q1 - 1.5 * IQR)) | (Wave_Height__Period > (Q3 + 1.5 * IQR)))]
print(df_no_outliers)
df_no_outliers.to_csv('data_03.csv')
df.describe().to_csv('data_features_01.csv')
df_no_outliers.describe().to_csv('data_features_02.csv')
# print(df.shape)







