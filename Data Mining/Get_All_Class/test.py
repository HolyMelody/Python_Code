import pandas as pd

# 读取数据
data = pd.read_csv('data.csv')

# 将日期时间数据转换为日期时间格式
data['Measurement_Date_And_Time'] = pd.to_datetime(data['Measurement_Date_And_Time'])
# 提取月份信息并存储为新列
data['Month'] = data['Measurement_Date_And_Time'].dt.month
#存为新的csv文件
data.to_csv('data_month.csv',index=False)
#检测，Month列是否有NaN值,有则将该行删除
print(data['Month'].isnull().any())
data = data.dropna(subset=['Month'])
print(data['Month'].isnull().any())
print(data['Month'].describe())
# 显示包含新列的数据
#print(data)