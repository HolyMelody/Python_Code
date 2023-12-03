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

# def Box_Plot(dataFrame):
    
#     # 创建一个指定大小的图像
#     plt.figure(figsize=(10, 6))

#     # 绘制盒图
#     columns_to_process = ['Wave_Period']
#     sb.boxplot(dataFrame[columns_to_process],color='skyblue',flierprops=dict(markerfacecolor='r', marker='s'))
#     # 设置y轴标签
#     plt.ylabel("Data (inches)")

#     # 显示盒图
#     plt.show()





# 读取CSV文件
df = pd.read_csv('data.csv')

# Box_Plot(df)
# 指定需要计算均值和填充的列
columns_to_process = ['Water_Temperature', 'Turbidity', 'Wave_Height','Wave_Period','Transducer_Depth','Battery_Life']
# 计算均值
mean_values = df[columns_to_process].\
apply(lambda x: np.mean(x[x >= 0]), axis=0)
# 填充负值和NaN值
df[columns_to_process] = df[columns_to_process].\
apply(lambda x: x.mask((x < 0) | x.isna(), mean_values[x.name]), axis=0)
# 将结果保存回原CSV文件
df.to_csv('Data.csv', index=False)
X=[]
Y=[]
X = np.array(df[columns_to_process])
Y = np.array(df['Beach_Name'])
train_accuracy, test_accuracy=train_and_evaluate_decision_tree(X, Y)
print("训练集准确率：", train_accuracy)
print("测试集准确率：", test_accuracy)