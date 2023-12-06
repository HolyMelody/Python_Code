import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

def visualize_missing_and_outliers(data, features):
    data = data.copy()
    # 根据沙滩名分组
    grouped_data = data.groupby('Beach_Name')
    # 可视化每个沙滩的缺失值和异常值
            # 可视化缺失值
    plt.figure(figsize=(8, 6))

    msno.matrix(data)
    plt.show()
            # 可视化异常值（负数）
            # plt.figure()
            # beach_data[feature][beach_data[feature] < 0].plot(kind='bar')
            # plt.title(f'{beach_name} - {feature} 异常值（负数）情况')
            # plt.show()

# 读取数据
data = pd.DataFrame({
    'Beach_Name': ['Montrose Beach', 'Ohio Street Beach', 'Calumet Beach', 'Calumet Beach'],
    'Measurement_Date_And_Time': ['2013-08-30T08:00:00', '2016-05-26T13:00:00', '2013-09-03T16:00:00', '2014-05-28T12:00:00'],
    'Water_Temperature': [20.3, 14.4, 23.2, 16.2],
    'Turbidity': [1.18, 1.23, 3.63, 1.26],
    'Transducer_Depth': [0.891, None, 1.201, 1.514],
    'Wave_Height': [0.08, 0.111, 0.174, 0.147],
    'Wave_Period': [3, 4, 6, 4],
    'Battery_Life': [9.4, 12.4, 9.4, 11.7],
    'Measurement_ID': ['MontroseBeach201308300800', 'OhioStreetBeach201605261300', 'CalumetBeach201309031600', 'CalumetBeach201405281200']
})

# 属性列表
features = ['Water_Temperature', 'Turbidity', 'Transducer_Depth', 'Wave_Height', 'Wave_Period', 'Battery_Life']

# 调用函数进行可视化
visualize_missing_and_outliers(data, features)