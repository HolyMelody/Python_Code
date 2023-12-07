#数据处理
import numpy as np
import pandas as pd
import math

# 决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
#交叉验证
from sklearn.model_selection import cross_validate
#决策树plus
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
#神经网络
import torch
import torch.nn as nn
import torch.optim as optim

#随机森林交叉验证
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

#盒图
import seaborn as sb
import matplotlib.pyplot as plt

#路径
import os

#可视化缺失值和负数值
import missingno as msno
#可视化离群值和缺失值
import missingno as msno
import seaborn as sns
#进度条
from tqdm import tqdm
import time

class Data_Processing():
    def __init__(self):
        pass

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
    def Data_As_Beach_Name(self,data,Features):
        data = data.copy()
        # 将DataFrame 数据按第一列的所有相同的Beach_Name排列，然后按照时间顺序排列(初始数据已经3按照时间排列)
        data = data.sort_values(by=['Beach_Name','Measurement_Date_And_Time'])
        #save as cvs
        data.to_csv('Data_as_beach_name.csv')
        data.groupby('Beach_Name').describe().to_csv('Data_as_beach_name_Features.csv')
        print("Yes,data as beach name complete!")
        return data

    def Visualize_Missing_and_Outliers(self,data, features):
        data_copy = data.copy()
        # 使用agg函数计算缺失值和负数值的数量
        def missing_count(x):
            return x.isnull().sum()
        def negative_count(x):
            return (x < 0).sum()
        # 使用agg函数计算缺失值和负数值的数量
        results = {}
        for feature in features:
            result = data.groupby("Beach_Name").agg({
                feature: [missing_count, negative_count]
            })
            results[feature] = result

        # 保留各Beach_Name的各特征的总和缺失值和负数值的数量
        beach_names = list(data["Beach_Name"].unique())
        missing_counts = {beach: sum(results[feature].loc[beach, (feature, 'missing_count')] for feature in features) for beach in beach_names}
        negative_counts = {beach: sum(results[feature].loc[beach, (feature, 'negative_count')] for feature in features) for beach in beach_names}

        # 创建柱状图
        fig, ax = plt.subplots()
        width = 0.35  # 柱状图宽度

        # 设置柱状图位置
        x = range(len(beach_names))
        # 设置字体的属性
        # plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
        plt.rcParams["font.sans-serif"] = "SimHei"
        plt.rcParams["axes.unicode_minus"] = False
        # 绘制missing_count柱状图
        rects1 = ax.bar(x, missing_counts.values(), width, label='各属性missing_count总和')
        # 绘制negative_count柱状图
        rects2 = ax.bar([i + width for i in x], negative_counts.values(), width, label='各属性negative_count总和')

        # 添加标签、标题和图例
        ax.set_ylabel('数量')
        ax.set_xlabel('Beach_Name')
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(beach_names)
        ax.legend()

        # 显示柱状图
        plt.show()


    # Define a function to calculate the count of outliers
    def Calculate_Outliers_Count(self,x):
        lower_bound = x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))
        upper_bound = x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))
        return ((x < lower_bound) | (x > upper_bound)).sum()

    # 可视化离群值和缺失值
    def Visualize_Outliers(self,data,Features):
        data= data.copy()
        # Features = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth','Battery_Life']
        # outliers_count_by_beach = data.groupby("Beach_Name")[Features]\
        # .apply(self.Calculate_Outliers_Count)

        Features = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth','Battery_Life']
        outliers_count = data[Features]\
        .apply(self.Calculate_Outliers_Count)
        #print(grouped)
        #plt.figure(figsize=(10, 6))  # 设置图像的大小为宽度10英寸，高度6英寸
        # 创建柱状图
        plt.bar(Features, outliers_count, color='green')
        # 设置字体的属性
        # plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
        plt.rcParams["font.sans-serif"] = "SimHei"
        plt.rcParams["axes.unicode_minus"] = False
        # 添加标题和标签
        plt.title('每个特征的异常值数量')
        plt.xlabel('特征')
        plt.ylabel('异常值数量')

        plt.xticks(rotation=15)  # 将横坐标的标签旋转45度
        # 显示柱状图
        plt.show()
        
        # outliers_count_by_beach.plot(kind="bar", stacked=True)
        # plt.xlabel("Beach_Name")
        # plt.ylabel("Count")
        # plt.title("Missing and Negative Values by Beach")
        # plt.legend(["Missing Values", "Negative Values"])
        # plt.xticks(rotation=15)  # 将横坐标的标签旋转45度
        # plt.show()
    # 绘制热图
    def Visualize_Correlation(self,data, column_to_analyze, features_to_process):
        # 计算相关性
        correlation_matrix = data[features_to_process].corr()

        # 选择要分析的列
        selected_correlation = correlation_matrix[column_to_analyze]
        
        # 设置字体的属性
        # plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
        plt.rcParams["font.sans-serif"] = "SimHei"
        plt.rcParams["axes.unicode_minus"] = False
        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('相关性热图')
        plt.show()
        print("热图已经绘制完毕！")
    
    def Data_Padding_As_Beach_name(self, data, Features,Fill_Way='Deficiency'):#'Negative''Quartile'
        data = data.copy()
        columns_to_process = Features
        
        # 填充负数值和缺失值
        if Fill_Way == 'Deficiency':
            for column in columns_to_process:
                # Fill missing and negative values with the mean of non-negative values for each beach
                filled_values = data.groupby('Beach_Name')[column].apply(lambda x: x.mask(x < 0, x[x >= 0].mean()).fillna(x[x >= 0].mean())).reset_index(level=0, drop=True)
                data[column] = filled_values
            print("负数值和缺失值已经填充完毕！")
        ##只填充缺失值
        elif Fill_Way == 'Negative':
            for column in columns_to_process:
                # 使用每个海滩的非负值的平均值来填充缺失值
                filled_values = data.groupby('Beach_Name')[column].apply(lambda x: x.fillna(x[x >= 0].mean())).reset_index(level=0, drop=True)
                data[column] = filled_values 
            print("缺失值已经填充完毕！")
        else:
            print("填充方式错误！")
        data_padded = pd.DataFrame(data)
        print("是的，按照'Beach_Name'列进行数据填充完成！")
        #.to_csv('Data_padding_as_beach_name.csv')
        data_padded.to_csv('Data_padding_as_beach_name.csv',index=False)
        
        return data_padded
    

    
    #进一步处理离群值
    def Data_Drop_Qartile(self, data, Features):
        features = Features
        data = self.Data_Padding_As_Beach_name(data, features, 'Deficiency')
        q1 = data.groupby('Beach_Name')[features].transform('quantile', 0.25)
        q3 = data.groupby('Beach_Name')[features].transform('quantile', 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (data[features] < lower_bound) | (data[features] > upper_bound)
        filled_values = data[features].where(~outliers, data.groupby('Beach_Name')[features].transform('mean'))
        processed_data = data.copy()
        processed_data[features] = filled_values
        processed_data.to_csv('processed_data.csv')
        print("是的，离群值已经处理完毕！")
        return processed_data
    #将填充结束后的数据装换为训练接口
    def Return_Port(self,data,Features):
        columns_to_process = Features
        X=[]
        Y=[]
        X = np.array(data[columns_to_process])
        Y = np.array(data['Beach_Name'])
        return X,Y

class Data_Classification:
    def __init__(self):
        pass

    def Start_Classification(self,Data_padded,features):
        #Return_Port
        Data_padding_as_beach_name=Data_padded
        Features = features 
        X,y = Data_Processing().Return_Port(Data_padding_as_beach_name,Features)
        for i in tqdm(range(10)):
            #time.sleep(0.1)  # 模拟训练过程
            train_accuracy, test_accuracy= self.\
            Random_Forest_Classification(X, y)
            #Train_and_Evaluate_Neural_Network(X, y)
            #train_and_evaluate_decision_tree(X,y)


        print("训练集准确率：{:.2f}%".format(train_accuracy * 100))
        print("测试集准确率：{:.2f}%".format(test_accuracy * 100))
        #print("交叉验证准确率：{:.2f}%".format(mean_cv_accuracy * 100))


    def train_and_evaluate_decision_tree(self,X, y):
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
        clf = DecisionTreeClassifier(max_depth= 9)

        # 使用训练集数据拟合模型
        decision_tree = clf.fit(X_train, y_train)

        # 使用交叉验证计算模型的准确率
        cv_scores = cross_val_score(decision_tree, X_train, y_train, cv=5)
        mean_cv_accuracy = np.mean(cv_scores)

        # 使用训练集数据进行预测
        y_train_pred = clf.predict(X_train)

        # 使用测试集数据进行预测
        y_test_pred = clf.predict(X_test)

        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

        print("决策树模型训练！")
  
        return train_accuracy, test_accuracy,mean_cv_accuracy
        # 返回训练集和测试集的准确率
        # surfing_preferences = {'Water_Temperature': 20, 'Turbidity': 5, 'Wave_Height': 2, 'Wave_Period': 10}
        # preferred_features = [surfing_preferences['Water_Temperature'], surfing_preferences['Turbidity'], surfing_preferences['Wave_Height'], surfing_preferences['Wave_Period']]
        # predicted_result = clf.predict([preferred_features])
        # print("根据您的偏好，推荐的沙滩是：", predicted_result[0])

    def Random_Forest_Classification(self,X, y):

        # 拆分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 创建和训练随机森林模型
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        # 预测并计算准确率
        y_train_pred = rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("随机森林模型训练！")
        return train_accuracy, test_accuracy
    # 定义神经网络模型神经网络函数名：train_and_evaluate_neural_network
    def Train_and_Evaluate_Neural_Network(self,X, y):
        """
        定义神经网络模型
        :param X: 训练数据
        :param y: 训练标签
        :return:
        """
        # 定义神经网络模型
        model = nn.Sequential(
            nn.Linear(X.shape[1], 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 6)  # 这里假设输出是二分类，如果是其他类别数量，需要调整
        )
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        # 定义优化器
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        # 将数据转换为张量
        X = torch.tensor(X, dtype=torch.float32)
        # 将字符串标签转换为整数形式的类别标签
        label_to_index = {label: index for index, label in enumerate(set(y))}
        y = torch.tensor([label_to_index[elem] for elem in y], dtype=torch.long)
        # 划分训练集和测试集
        train_size = int(0.7 * X.shape[0])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        # 开始训练
        for epoch in range(2000):
            # 前向传播
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印训练过程中的损失
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{2000}], Loss: {loss.item()}')
        # 计算训练集和测试集的准确率
        with torch.no_grad():
            train_outputs = model(X_train)
            _, train_predicted = torch.max(train_outputs, 1)
            train_accuracy = (train_predicted == y_train).float().mean()
            test_outputs = model(X_test)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = (test_predicted == y_test).float().mean()
        return train_accuracy.item(), test_accuracy.item()



class Property_Metrics:
    def __init__(self,Original_Data):
        #冲浪英文为surfing
        Features_Surfing = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period']
        Data_processing = Data_Processing()
        self.Data_Default = Data_processing.Data_Drop_Qartile(Original_Data,Features_Surfing)
    def Metrics_Rules_Set(self,weight):
        data=self.Data_Default
        #遍历'Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period',
        Data_Rules = data.copy()
        print(weight)
        print(weight[0])
        #print(weight[0].dtype)
        print(Data_Rules['Water_Temperature'].dtype)
        #定义一个规则
        Data_Rules['Metrics'] = Data_Rules['Water_Temperature']+\
            Data_Rules['Turbidity']+\
            100*Data_Rules['Wave_Height']+\
            Data_Rules['Wave_Period']
        Data_Rules['Metrics_get'] = weight[0]*Data_Rules['Water_Temperature']+\
            weight[1]*Data_Rules['Turbidity']+\
            weight[2]*100*Data_Rules['Wave_Height']+\
            weight[3]*Data_Rules['Wave_Period']
        Data_Rules['Is_Exception'] = np.where(Data_Rules['Metrics']>100,1,0)
        Data_Rules.to_csv('Data_Rules.csv')
        Data_Rules.groupby('Beach_Name')['Metrics'].describe().to_csv('Data_Rules_Features.csv')
        print("Yes,Data_Rules_Set complete!")
        return Data_Rules

    def User_Information_Get(self):
        weights = [] 
        print("*****************************************")
        print("欢迎使用冲浪指数查询系统！")
        print("*****************************************")
        level = input("你是冲浪新手吗？(y/n): ")
        if level == 'y':
            '''
            数值尽量
            '''
            temperature = input("请输入你需要的大概水温(出于安全,请在17-22之间)")#9.1-29.6
            turbidity = input("请输入你需要的大概浑浊度(出于安全,请在0-5之间)")#0.01-14.8
            wave_height = input("请输入你需要的大概波浪高度(出于安全,请在11-19之间)")#1.3-320
            wave_period = input("请输入你需要的大概波浪周期(出于安全,请在3-4之间)")#1-8
            weights = [float(-temperature),float(-turbidity),float(-wave_height),float(wave_period)]
        elif level == 'n':
            temperature = input("请输入你需要的大概水温(出于体验,推荐17以下之间)")#9.1-29.6
            turbidity = input("请输入你需要的大概浑浊度(出于体验,请在0-5之间)")#0.01-14.8
            wave_height = input("请输入你需要的大概波浪高度(出于体验,请在11-19之间)")#1.3-320
            wave_period = input("请输入你需要的大概波浪周期(出于体验,请在3-4之间)")#1-8
            weights = [float(-temperature),float(turbidity),float(wave_height),float(wave_period)]
        else:
            print("输入错误！")
        self.Metrics_Rules_Set(weights)
        print(weights == [20,5,15,4])
        print("信息已经录入完毕！")
#主函数
if __name__ == "__main__":
    #从本地读取数据
    # script_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_path, "data.csv")
    Original_Data = pd.read_csv('data.csv')  
    Features_to_process = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth','Battery_Life']
    Features = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth']
    #数据处理
    Data_processing = Data_Processing()
    #Data_as_beach_name = Data_Processing.Data_As_Beach_Name(Original_Data)
    #Data_Processing.Visualize_Missing_and_Outliers(Original_Data,Features_to_process)
    Data_padding_as_beach_name = Data_processing.Data_Padding_As_Beach_name(Original_Data,Features_to_process,Fill_Way='Negative')
    #Data_drop_qartile = Data_Processing.Data_Drop_Qartile(Original_Data,Features_to_process)
    #沙滩分类
    #Start_Classification
    # Data_classification = Data_Classification()
    # Data_classification.Start_Classification(Data_padding_as_beach_name,Features)
    #Data_processing.Data_As_Beach_Name(Original_Data,Features_to_process)
    Features_Anasysis = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth','Battery_Life']
    Data_processing.Visualize_Correlation(Original_Data,'Transducer_Depth',Features_Anasysis)
    
    #冲浪指数
    # Property_metrics = Property_Metrics(Original_Data)
    # Property_metrics.User_Information_Get()
    #Data_Rules = Property_metrics.Metrics_Rules_Set(weights)
    #weight = [1,0.8,5,2]
    #Data_Rules = Property_metrics.Metrics_Rules(Data_drop_qartile,weight)
    #Data_drop_qartile.to_csv('Data_drop_qartile.csv')
    #Data_Rules.describe().to_csv('Data_Drop_Qartile_Features.csv')
    #print("Yes,Data_Rules complete!")
    # Data_Rules.groupby('Beach_Name')['Metrics'].describe().to_csv('Data_Rules_Features.csv')
    #可视化缺失值和负数值
    # visualize_missing_and_outliers = Data_Processing.\
    # Visualize_Missing_and_Outliers(Original_Data)
    #Data_Processing.Visualize_Outliers(Data_padding_as_beach_name,Features)

    #数据可视化 - 盒图
    #save to csv
    #Data_as_beach_name.to_csv('Data_as_beach_name.csv')
    # Paded_Features = ['Wave_Height','Wave_Period']
    # Paded_Data = Data_Processing.Data_Padding(Original_Data,Features)
    #Box_plot = Data_Processing.Box_Plot(data)
    # Time_plot = Data_Processing.Time_Plot(Original_Data,Paded_Features)
    # Time_plot = Data_Processing.Time_Plot(Paded_Data,Paded_Features)