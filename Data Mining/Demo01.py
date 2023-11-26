#数据集链接：https://datahub.io/JohnSnowLabs/beach-water-quality---automated-sensors
#本地数据集路径：E:\Utility ROOM\大三上\大数据挖掘\课设\数据挖掘课程数据集\天气自动传感器大数据（Beach Water Quality）挖掘\data.csv
# 数据说明：这个数据集统计了在芝加哥的若干个海滩中传感器每日测试的数据，这些数据包括日期、水温、浑浊度、
# 	 换能器深度(m)、波浪高度(m)、波浪周期(s)、传感器电池剩余情况这些属性。一共34924条数据。

#csv文件中的数据格式如下：
#Beach_Name	Measurement_Date_And_Time	Water_Temperature	Turbidity	Transducer_Depth	Wave_Height	Wave_Period	Battery_Life	Measurement_ID
#前五行格式如下：
# Beach_Name	Measurement_Date_And_Time	Water_Temperature	Turbidity	Transducer_Depth	Wave_Height	Wave_Period	Battery_Life	Measurement_ID
# Montrose Beach	2013-08-30T08:00:00	20.3	1.18	0.891	0.08	3	9.4	MontroseBeach201308300800
# Ohio Street Beach	2016-05-26T13:00:00	14.4	1.23		0.111	4	12.4	OhioStreetBeach201605261300
# Calumet Beach	2013-09-03T16:00:00	23.2	3.63	1.201	0.174	6	9.4	CalumetBeach201309031600
# Calumet Beach	2014-05-28T12:00:00	16.2	1.26	1.514	0.147	4	11.7	CalumetBeach201405281200

# 编写python程序完成以下任务：
#任务1：多分类。每个海滩的属性有差异，可以根据这些差异训练分类器完成对不同海滩的分类操作。
# 任务2：序列预测。对于某个沙滩，其水温、波浪高度、周期等属性可能会发生周期性变化。
#            可以根据过去几年观测到的数据预测接下来一段时间的情况。
# 任务3：聚类。给定测试数据，将其聚类，看看每一个类是否对应于一个海滩。
# 任务4：数据预处理。

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# 任务1：多分类。每个海滩的属性有差异，可以根据这些差异训练分类器完成对不同海滩的分类操作,用海滩名作标识。
# 分别选取水温(Water_Temperature)、浑浊度(Turbidity)、波浪高度(Wave_Height)、波浪周期(Battery_Life)
# 划分不同类型的海滩
#Input X contains NaN
class Beach_Classification:
    def __init__(self, data):
        self.data = data
        self.X = self.data[['Water_Temperature', 'Turbidity', 'Wave_Height', 'Wave_Period', 'Battery_Life']]
        self.y = self.data['Beach_Name']
    def cout_many(self):
        label_counts =self.y.value_counts()
        self.label_list = list(label_counts.index)
        # 打印每个标签和对应的重复次数
        for label, count in label_counts.items():
            print(f"标签 {label} 重复了 {count} 次")
    
    def beach_classification(self):
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, self.y, test_size=0.2, random_state=42)
        # Train a RandomForestClassifier
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = classifier.predict(X_test)
        # Evaluate the accuracy of the classifier
        classification_report_output = classification_report(y_test, y_pred, target_names=df['Beach_Name'].unique(), output_dict=True)
        num_classes = len(classification_report_output.keys()) - 3  # Subtracting 3 to exclude 'accuracy', 'macro avg', and 'weighted avg'

        print(f'Number of Classes: {num_classes}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=self.y.unique()))













data = pd.read_csv('E:/Utility ROOM/大三上/大数据挖掘/课设/数据挖掘课程数据集/天气自动传感器大数据（Beach Water Quality）挖掘/data.csv')
Beach_Classification = Beach_Classification(data)
Beach_Classification.cout_many()
Beach_Classification.beach_classification()


