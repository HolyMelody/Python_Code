from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.metrics import accuracy_score

def Dict_to_list(data_dict):
    X = []
    Y = []
    X_average ={}
    X_median = {}

    for key in data_dict:
        Remove_NaN = [[x] for x in data_dict[key] if not math.isnan(x)]
        X_average[key]=np.mean(Remove_NaN)
        X_median[key]=np.median(Remove_NaN)
        X=X+Remove_NaN
        Y=Y+[[key]]*len(Remove_NaN)

    # for value in data_dict[key]:
    #     if not math.isnan(value):
    #         X.append([value])
    #         Y.append([key])
    X = np.array(X)
    Y = np.array(Y)

    print("Yes,Dict_to_list!")
    return X,Y


def K_Means(data_dict):
    X, Y = Dict_to_list(data_dict)

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 创建并训练K-Means模型
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    # 在测试集上进行预测
    y_pred = kmeans.predict(X_test)

    # 打印预测结果
    print(y_pred)

    #准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

# 示例数据字典
data_dict = {'feature1': [1, 2, 3, 4, 5],
             'feature2': [6, 7, 8, 9, 10],
             'label': [0, 1, 0, 1, 0]}

# 调用K_Means函数
K_Means(data_dict)