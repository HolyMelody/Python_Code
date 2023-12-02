import pandas as pd
import numpy as np


# def Dict_to_list(self,data_dict):#用均值填充NaN
#     X = []
#     Y = []
#     X_average ={}
#     X_median = {}
#     for key,values in data_dict.items():
#         index_num=[]
#         X_Temp=[]
#         for index, element in enumerate(values):
#             if math.isnan(element):
#                 index_num.append(index)
#             else:
#                 X_Temp.append(element)
#         if X_Temp != []:
#             X_average[key]=np.mean(X_Temp)
#             X_median[key]=np.median(X_Temp)
#         else:
#             X_average[key]=0
#             X_median[key]=0
#         if index_num != []:#有NaN,马上替换
#             for index in index_num:
#                 values[index]=X_average[key] 

#         X.append(values)
#         Y.append([key]*len(values))

#     # X = np.array(X)
#     #Y = np.array(Y)
#     print("Yes,Dict_to_list complete!")
#     return X,Y

# def data_cleaning(data):
#     Water_Temperature_label_dict={}#水温
#     Turbidity_label_dict={}#浑浊度
#     Transducer_Depth_label_dict={}#换能器深度
#     Wave_Height_label_dict={}#波浪高度
#     Wave_Period_label_dict={}#波浪周期
#     Battery_Life_label_dict={}#传感器电池剩余情况
#     #seld.data的第一行


#     #将第三、四、六、七列的数据按照海滩名进行分组，用label_dict={}存储
#     for index,row in data.iterrows():#遍历每一行
#         Beach_Name_key = row.iloc[0]
#         #Measurement_Date_And_Time=row.iloc[1]
#         Water_Temperature_values = row.iloc[2]
#         Turbidity_label_values = row.iloc[3]
#         Transducer_Depth_values = row.iloc[4]
#         Wave_Height_values = row.iloc[5]
#         Wave_Period_values = row.iloc[6]
#         Battery_Life_values = row.iloc[7]

#         if Beach_Name_key in Water_Temperature_label_dict:
#             Water_Temperature_label_dict[Beach_Name_key].append(Water_Temperature_values)
#         else:
#             Water_Temperature_label_dict[Beach_Name_key]=[Water_Temperature_values]
        
#         if Beach_Name_key in Turbidity_label_dict:
#             Turbidity_label_dict[Beach_Name_key].append(Turbidity_label_values)
#         else:
#             Turbidity_label_dict[Beach_Name_key]=[Turbidity_label_values]

#         if Beach_Name_key in Transducer_Depth_label_dict:
#             Transducer_Depth_label_dict[Beach_Name_key].append(Transducer_Depth_values) 
#         else:
#             Transducer_Depth_label_dict[Beach_Name_key]=[Transducer_Depth_values]

#         if Beach_Name_key in Wave_Height_label_dict:
#             Wave_Height_label_dict[Beach_Name_key].append(Wave_Height_values)
#         else:
#             Wave_Height_label_dict[Beach_Name_key]=[Wave_Height_values]

#         if Beach_Name_key in Wave_Period_label_dict:
#             Wave_Period_label_dict[Beach_Name_key].append(Wave_Period_values) 
#         else:
#             Wave_Period_label_dict[Beach_Name_key]=[Wave_Period_values]

#         if Beach_Name_key in Battery_Life_label_dict:
#             Battery_Life_label_dict[Beach_Name_key].append(Battery_Life_values)
#         else: 
#             Battery_Life_label_dict[Beach_Name_key]=[Battery_Life_values]
#     print("Yes,data cleaning complete!")
#     return Water_Temperature_label_dict,\
#     Turbidity_label_dict,Transducer_Depth_label_dict,\
#     Wave_Height_label_dict,Wave_Period_label_dict,\
#     Battery_Life_label_dict


# df = pd.read_csv('data.csv')
# Water_Temperature_label_dict,\
# Turbidity_label_dict,Transducer_Depth_label_dict,\
# Wave_Height_label_dict,Wave_Period_label_dict,\
# Battery_Life_label_dict=data_cleaning(df)


#Wave_Height = pd.DataFrame.from_dict(Wave_Height_label_dict,orient='index')
#Wave_Height.to_csv('Wave_Height.csv')
#Wave_Height.apply(pd.Series.describe, axis=1).to_csv('Wave_Height_Features.csv')

#Wave_Period = pd.DataFrame.from_dict(Wave_Period_label_dict,orient='index')
# Wave_Period.to_csv('Wave_Period.csv')
# Wave_Period.apply(pd.Series.describe, axis=1).to_csv('Wave_Period_Features.csv')

# Wave_Height = Wave_Height.transpose()
# Wave_Period = Wave_Period.transpose()
# Wave_Height = Wave_Height[Wave_Height >= 0].dropna()
# Wave_Period = Wave_Period[Wave_Period >= 0].dropna()
# Wave_Height.to_csv('Wave_Height_no_negative.csv')
# Wave_Height.describe().\
# to_csv('Wave_Height_no_negative_features.csv')
# Wave_Period.to_csv('Wave_Period_no_negative.csv')
# Wave_Period.describe().\
# to_csv('Wave_Period_no_negative_features.csv')
# Wave_Height_describe=Wave_Height.describe()
# print(list(Wave_Height_describe.iloc[1,:]))
# if '63rd Street Beach' in Wave_Height_describe:
#     print(Wave_Height.describe().loc["mean","63rd Street Beach"])






# Wave_Height = Wave_Height.transpose()
# Wave_Height.to_csv('Wave_Height.csv')
# Q1 = Wave_Height.quantile(0.25)
# Q3 = Wave_Height.quantile(0.75)
# IQR = Q3 - Q1
# df_no_outliers_Wave_Height = \
# Wave_Height[~((Wave_Height < \
# (Q1 - 1.5 * IQR)) | (Wave_Height > (Q3 + 1.5 * IQR)))]
# df_no_outliers_Wave_Height.to_csv('Wave_Height_no_outliers.csv')
# df_no_outliers_Wave_Height.describe().\
# to_csv('Wave_Height_no_outliers_features.csv')


# Wave_Period = Wave_Period.transpose()
# Wave_Period.to_csv('Wave_Period.csv')
# Q1 = Wave_Period.quantile(0.25)
# Q3 = Wave_Period.quantile(0.75)
# IQR = Q3 - Q1
# df_no_outliers_Wave_Period = \
# Wave_Period[~((Wave_Period < \
# (Q1 - 1.5 * IQR)) | (Wave_Period > (Q3 + 1.5 * IQR)))]
# df_no_outliers_Wave_Period.to_csv('Wave_Period_no_outliers.csv')
# df_no_outliers_Wave_Period.describe().\
# to_csv('Wave_Period_no_outliers_features.csv')

# def Data_Padding(data_dict):#用均值填充NaN和负数
#     # 创建示例数据
#     # data = np.array([1, 2, -1, 3, np.nan, 5])
#     # # 创建掩码数组
#     # mask = np.ma.masked_where((data < 0) | np.isnan(data), data)
#     # # 计算平均值
#     # mean_value = np.ma.mean(mask)
#     # print(mean_value)
#         for key,values in data_dict.items():
#             for index, element in enumerate(values):
#                 element = np.array(element)
#                 mask = np.ma.masked_where((element < 0) | np.isnan(element), element)



