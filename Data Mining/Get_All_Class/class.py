#数据处理
import numpy as np
import pandas as pd

# 决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

#盒图
import seaborn as sb
import matplotlib.pyplot as plt

#路径
import os



class Data_Processing:
    def __init__(self, data):
        self.data = data.copy()

    #先行按沙滩名分类
    def data_cleaning(self,data):
        data = data.copy()
        Water_Temperature_label_dict={}#水温
        Turbidity_label_dict={}#浑浊度
        Transducer_Depth_label_dict={}#换能器深度
        Wave_Height_label_dict={}#波浪高度
        Wave_Period_label_dict={}#波浪周期
        Battery_Life_label_dict={}#传感器电池剩余情况
        #seld.data的第一行


        #将第三、四、六、七列的数据按照海滩名进行分组，用label_dict={}存储
        for index,row in data.iterrows():#遍历每一行
            Beach_Name_key = row.iloc[0]
            #Measurement_Date_And_Time=row.iloc[1]
            Water_Temperature_values = row.iloc[2]
            Turbidity_label_values = row.iloc[3]
            Transducer_Depth_values = row.iloc[4]
            Wave_Height_values = row.iloc[5]
            Wave_Period_values = row.iloc[6]
            Battery_Life_values = row.iloc[7]

            if Beach_Name_key in Water_Temperature_label_dict:
                Water_Temperature_label_dict[Beach_Name_key].append(Water_Temperature_values)
            else:
                Water_Temperature_label_dict[Beach_Name_key]=[Water_Temperature_values]
            
            if Beach_Name_key in Turbidity_label_dict:
                Turbidity_label_dict[Beach_Name_key].append(Turbidity_label_values)
            else:
                Turbidity_label_dict[Beach_Name_key]=[Turbidity_label_values]

            if Beach_Name_key in Transducer_Depth_label_dict:
                Transducer_Depth_label_dict[Beach_Name_key].append(Transducer_Depth_values) 
            else:
                Transducer_Depth_label_dict[Beach_Name_key]=[Transducer_Depth_values]

            if Beach_Name_key in Wave_Height_label_dict:
                Wave_Height_label_dict[Beach_Name_key].append(Wave_Height_values)
            else:
                Wave_Height_label_dict[Beach_Name_key]=[Wave_Height_values]

            if Beach_Name_key in Wave_Period_label_dict:
                Wave_Period_label_dict[Beach_Name_key].append(Wave_Period_values) 
            else:
                Wave_Period_label_dict[Beach_Name_key]=[Wave_Period_values]

            if Beach_Name_key in Battery_Life_label_dict:
                Battery_Life_label_dict[Beach_Name_key].append(Battery_Life_values)
            else: 
                Battery_Life_label_dict[Beach_Name_key]=[Battery_Life_values]
        print("Yes,data cleaning complete!")
        return Water_Temperature_label_dict,\
        Turbidity_label_dict,Transducer_Depth_label_dict,\
        Wave_Height_label_dict,Wave_Period_label_dict,\
        Battery_Life_label_dict
    
    #数据可视化 - 盒图
    def Box_Plot(self,data,Features):
        data = data.copy()
        columns_to_process = Features
        # 创建一个指定大小的图像
        plt.figure(figsize=(10, 6))
        # 绘制盒图
        
        sb.boxplot(data[columns_to_process],color='skyblue',flierprops=dict(markerfacecolor='r', marker='s'))
        # 设置y轴标签
        plt.ylabel("Data (inches)")
        # 显示盒图
        plt.show()

    # 按沙滩名绘制时间-数据序列图
    def Time_Plot(self,data,features):
        data = data.copy()
        columns_to_process = features  #columns_to_process = ['Wave_Height', 'Wave_Period']
        Beach_Name = ['Montrose Beach', 'Ohio Street Beach',
                    'Calumet Beach', '63rd Street Beach', 'Osterman Beach', 'Rainbow Beach']

        plt.figure(figsize=(10, 6))  # Create a single figure

        line_styles = ['-', '--', '-.', ':', '-.', '--']  # Define line styles
        line_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Define line colors

        for i, beach_name in enumerate(Beach_Name):
            dataFrame = data.loc[data['Beach_Name'] == beach_name]
            dataFrame['Measurement_Date_And_Time'] = pd.to_datetime(dataFrame['Measurement_Date_And_Time'])
            dataFrame['月份'] = dataFrame['Measurement_Date_And_Time'].dt.strftime('%Y-%m')
            monthly_data = dataFrame.groupby('月份')[columns_to_process].mean()

            plt.plot(monthly_data.index, monthly_data['Wave_Height'], linestyle=line_styles[i], color=line_colors[i], label=beach_name + ' Wave_Height')
            plt.plot(monthly_data.index, monthly_data['Wave_Period'], linestyle=line_styles[i], color=line_colors[i], label=beach_name + ' Wave_Period')

        plt.title('Time_Data')
        plt.xlabel('Month')
        plt.ylabel('Month_Mean_Data')
        plt.legend()
        plt.show()

    #输入label_dict，按沙滩名描述各类数据
    def Data_Describe(self,label_dict):#待修改
        Wave_Height_label_dict, Wave_Period_label_dict = {}, {}#Wave_Height_label_dict
        Wave_Height = pd.DataFrame.from_dict(Wave_Height_label_dict,orient='index')
        Wave_Height.to_csv('Wave_Height.csv')
        Wave_Height.apply(pd.Series.describe, axis=1).to_csv('Wave_Height_Features.csv')

        Wave_Period = pd.DataFrame.from_dict(Wave_Period_label_dict,orient='index')
        Wave_Period.to_csv('Wave_Period.csv')
        Wave_Period.apply(pd.Series.describe, axis=1).to_csv('Wave_Period_Features.csv')

    # 填充负值和NaN值
    def Data_Padding(self,data,Features):
        data = data.copy()
        columns_to_process = Features
        # 计算均值
        mean_values = data[columns_to_process].\
        apply(lambda x: np.mean(x[x >= 0]), axis=0)
        # 填充负值和NaN值
        data[columns_to_process] = data[columns_to_process].\
        apply(lambda x: x.mask((x < 0) | x.isna(), mean_values[x.name]), axis=0)
        # 将结果保存为dataframe,并返回
        data_paded = pd.DataFrame(data)
        print("Yes,data padding complete!")
        return data_paded
    
    #将数据按沙滩名排列
    def Data_As_Beach_Name(self,data):
        data = data.copy()
        # 将DataFrame 数据按第一列的所有相同的Beach_Name排列，然后按照时间顺序排列(初始数据已经3按照时间排列)
        data = data.sort_values(by=['Beach_Name','Measurement_Date_And_Time'])
        print("Yes,data as beach name complete!")
        return data
    def Data_Padding_As_Beach_name(self, data, Features):
        data = data.copy()
        columns_to_process = Features
        
        # 按照'Beach_Name'列进行分组，并计算每个分组的均值
        mean_values = data.groupby('Beach_Name')[columns_to_process].transform('mean')
        
        # 填充负值和NaN值
        data[columns_to_process] = data[columns_to_process].mask((data[columns_to_process] < 0) | data[columns_to_process].isna(), mean_values)
        
        # 将结果保存为DataFrame，并返回
        data_padded = pd.DataFrame(data)
        print("是的，按照'Beach_Name'列进行数据填充完成！")
        return data_padded
    
    #将填充结束后的数据装换为训练接口
    def Return_Port(self,data,Features):
        columns_to_process = Features
        X=[]
        Y=[]
        X = np.array(data[columns_to_process])
        Y = np.array(data['Beach_Name'])
        return X,Y

class Data_Classification():
    def __init__(self, data):
        pass


    def train_and_evaluate_decision_tree(X, y):
        """
        使用输入的特征数据和目标数据训练决策树模型，并计算训练集和测试集的准确率

        参数：
        X: 特征数据
        y: 目标数据

        返回：
        训练集和测试集的准确率
        """
        # 将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 实例化决策树分类器
        clf = DecisionTreeClassifier()

        # 使用训练集数据拟合模型
        clf.fit(X_train, y_train)

        # 使用训练集数据进行预测
        y_train_pred = clf.predict(X_train)

        # 使用测试集数据进行预测
        y_test_pred = clf.predict(X_test)

        # 计算训练集和测试集的准确率
        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

        # 返回训练集和测试集的准确率
        return train_accuracy, test_accuracy
    

#主函数
if __name__ == "__main__":
    #从本地读取数据
    # script_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_path, "data.csv")
    Original_Data = pd.read_csv('data.csv')  
    Features = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth','Battery_Life']
    Data_Processing = Data_Processing(Original_Data)
    Data_as_beach_name = Data_Processing.Data_As_Beach_name(Original_Data)
    Data_padding_as_beach_name = Data_Processing.Data_Padding_As_Beach_name(Original_Data,Features)
    Data_padding_as_beach_name.to_csv('Data_padding_as_beach_name.csv')
    #save to csv
    #Data_as_beach_name.to_csv('Data_as_beach_name.csv')
    # Paded_Features = ['Wave_Height','Wave_Period']
    # Paded_Data = Data_Processing.Data_Padding(Original_Data,Features)
    #Box_plot = Data_Processing.Box_Plot(data)
    # Time_plot = Data_Processing.Time_Plot(Original_Data,Paded_Features)
    # Time_plot = Data_Processing.Time_Plot(Paded_Data,Paded_Features)