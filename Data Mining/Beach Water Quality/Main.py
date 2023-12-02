#数据
import numpy as np
import pandas as pd

#代码进度条
from tqdm import tqdm
import time
#判断NaN
import math

#路径
import os

#机器学习数据格式
from sklearn.utils import column_or_1d


#支持向量机
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#K-Means
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

#网格搜索，优化超参数
from sklearn.model_selection import GridSearchCV


#先对数据进行清洗
class Data_Processing:
    def __init__(self,data):
        self.data = data
        #字典key为沙滩名，value为数据名，按列处理
        #----------------------------------------------------
        self.Water_Temperature_label_dict,\
        self.Turbidity_label_dict,\
        self.Transducer_Depth_label_dict,\
        self.Wave_Height_label_dict,\
        self.Wave_Period_label_dict,\
        self.Battery_Life_label_dict = self.data_cleaning()
        print("Yes,Data_Processing init complete!")
        #----------------------------------------------------
#--------0  -----------------------1  ----------------2
#Beach_Name,Measurement_Date_And_Time,Water_Temperature,
#--------3  -------------3  --------4   ----------5  ----------6 -----------7
#Turbidity,Transducer_Depth,Wave_Height,Wave_Period,Battery_Life,Measurement_ID

    #日期、水温、浑浊度、
    #换能器深度(m)、波浪高度(m)、波浪周期(s)、传感器电池剩余情况
    #字典key为沙滩名，value为数据名，按列处理
    def Receive_data_cleaning(self,data_name):
            
        mapping={'Water_Temperature':self.Water_Temperature_label_dict,\
                'Turbidity':self.Turbidity_label_dict,\
                'Transducer_Depth':self.Transducer_Depth_label_dict,\
                'Wave_Height':self.Wave_Height_label_dict,\
                'Wave_Period':self.Wave_Period_label_dict,\
                'Battery_Life':self.Battery_Life_label_dict\
                }
        return mapping[data_name]
    def data_cleaning(self):
        Water_Temperature_label_dict={}#水温
        Turbidity_label_dict={}#浑浊度
        Transducer_Depth_label_dict={}#换能器深度
        Wave_Height_label_dict={}#波浪高度
        Wave_Period_label_dict={}#波浪周期
        Battery_Life_label_dict={}#传感器电池剩余情况
        #seld.data的第一行
  

        #将第三、四、六、七列的数据按照海滩名进行分组，用label_dict={}存储
        for index,row in self.data.iterrows():#遍历每一行
            Beach_Name_key = row.iloc[0]
            Measurement_Date_And_Time=row.iloc[1]
            Water_Temperature_values = row.iloc[2]
            Turbidity_label_values = row.iloc[3]
            Transducer_Depth=row.iloc[4]
            Transducer_Depth_values = row.iloc[5]
            Wave_Height_values = row.iloc[6]
            Wave_Period_values = row.iloc[7]
            Battery_Life_values = row.iloc[8]

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
    def All_Features_values(self,data_dict):
        X_average ={}
        X_median = {}
        for key,values in data_dict.items():
            index_num=[]
            X_Temp=[]
            for index, element in enumerate(values):
                if math.isnan(element):
                    index_num.append(index)
                else:
                    X_Temp.append(element)
            if X_Temp != []:
                X_average[key]=np.mean(X_Temp)
                X_median[key]=np.median(X_Temp)
            else:
                X_average[key]=0
                X_median[key]=0
            if index_num != []:#有NaN,马上替换
                for index in index_num:
                    values[index]=X_average[key] 
        print("Yes,All_Features_values complete!")
        return [X_average,X_median]
    
    def Dict_to_list(self,data_dict):#用均值填充NaN
        X = []
        Y = []
        X_average ={}
        X_median = {}
        for key,values in data_dict.items():
            index_num=[]
            X_Temp=[]
            for index, element in enumerate(values):
                if math.isnan(element):
                    index_num.append(index)
                else:
                    X_Temp.append(element)
            if X_Temp != []:
                X_average[key]=np.mean(X_Temp)
                X_median[key]=np.median(X_Temp)
            else:
                X_average[key]=0
                X_median[key]=0
            if index_num != []:#有NaN,马上替换
                for index in index_num:
                    values[index]=X_average[key] 

            X.append(values)
            Y.append([key]*len(values))

        # X = np.array(X)
        #Y = np.array(Y)
        print("Yes,Dict_to_list complete!")
        return X,Y
    def Multidimensional_Features_list(self,data_dict_list):
        X = []
        Y = []
        for dictionary in data_dict_list:
            X_temp,Y_temp=self.Dict_to_list(dictionary)
            X.append(X_temp)
            Y.append(Y_temp)
        X=np.hstack(X)
        Y=np.hstack(Y)
        Y = np.array([list(set(row)) for row in Y])
        print("Yes,Multidimensional_Features_list!")
        return X,Y
    def Mul_Features_list_second_edtion(self,Data_Name):
        X = []
        Y = []
        for data_name in Data_Name:
            mean=self.All_Features_values(self.Receive_data_cleaning(data_name))[0]
            Mean=[round(x,1) for x in mean.values()]
        print(Mean)
        for index in range(len(self.data)):
            X_temp=[]
            for i,data_name in enumerate(Data_Name):
                if not math.isnan(self.data.loc[index][data_name]):
                    X_temp.append(self.data.loc[index][data_name])
                else :
                    X_temp.append(Mean[i])
            X.append(X_temp)
            Y.append([self.data.loc[0]["Beach_Name"]])


        X=np.array(X)
        Y=np.array(Y)
        print(X[0:10])
        print(Y[0:10])
        print("Yes,Mul_Features_list_second_edtion complete!")



class Classification:
    def __init__(self, data):
        self.inner=self.Class_Method()
        print("Yes,Classification init complete!")

    # 计算每个标签的重复次数
    def cout_many(self):
        label_counts =self.y.value_counts()
        self.label_list = list(label_counts.index)
        # 打印每个标签和对应的重复次数
        for label, count in label_counts.items():
            print(f"标签 {label} 重复了 {count} 次")



    class Class_Method:
        def __init__(self):
            pass

        def SVM(self, data_dict):
            X, Y = data_dict
            # 将数据集拆分为训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # 创建进度条对象
            progress_bar = tqdm(total=len(X_train), desc='Training', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

            # 创建SVM分类器对象，使用线性内核
            clf = svm.SVC(kernel='linear')

            for i in range(len(X_train)):
                time.sleep(0.01)

                # 在训练集上训练模型
                clf.fit(X_train, y_train)

                # 更新进度条的后缀信息
                progress_bar.set_postfix({"Iteration": i+1})
                progress_bar.update(1)

            # 关闭进度条
            progress_bar.close()

            y_pred = clf.predict(X_test)

            # 定义超参数的候选值，缩小搜索范围
            param_grid = {
                'C': [0.1, 1],
                'gamma': [0.1, 0.01],
            }

            # 使用GridSearchCV进行超参数调优
            grid_search = GridSearchCV(clf, param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            # 输出最佳参数和对应的准确率
            print("最佳参数：", grid_search.best_params_)
            print("最佳准确率：", grid_search.best_score_)

            print("Yes, SVM!")
        
        #K-Means聚类
        def K_Means(self, data_dict):
            X, Y = data_dict

            # 将数据集拆分为训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            #创建进度条对象
            progress_bar = tqdm(total=len(X_train), desc='Training', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

            # 创建KMeans对象并进行聚类
            kmeans = MiniBatchKMeans(n_clusters=len(data_dict), n_init=3)
            kmeans.fit(X_train)

            for i in range(len(X_train)):
                time.sleep(0.01)

                # 更新进度条的后缀信息
                progress_bar.set_postfix({"Iteration": i + 1})
                progress_bar.update(1)

                # 在训练集上训练模型
                kmeans.partial_fit([X_train[i]])

            # 关闭进度条
            progress_bar.close()

            # 在测试集上进行预测
            y_pred = kmeans.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            print("准确率：", accuracy)
            print("Yes, K_Means!")





class Work_pcoessing:
    def __init__(self,data):
        self.data = data
        self.X = self.data[['Water_Temperature', 'Turbidity', 'Wave_Height', 'Wave_Period', 'Battery_Life']]
        self.y = self.data['Beach_Name']
        self.Data_processing=Data_Processing(self.data)
    def beach_classification(self):
        My_Classification=Classification(self.data)
        # My_Class_Method_SVM=My_Classification.inner.\
        # SVM(self.Water_Temperature_label_dict)
        #Data_Features_dict = self.Data_processing.Dict_to_list(self.Water_Temperature_label_dict)
        self.Data_processing.Mul_Features_list_second_edtion(['Water_Temperature','Turbidity'])
        # My_Class_Method_K_Means=My_Classification.inner.\
        # K_Means(Data_Features_dict)



#主函数
if __name__ == "__main__":
    #从本地读取数据
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_path, "data.csv")
    data = pd.read_csv(file_path)  
    Work_Pcoessing = Work_pcoessing(data)
    Work_Pcoessing.beach_classification()
    #编写代码进度条
    #单独用一维数据聚类准确度，非常低，需要升级维度









