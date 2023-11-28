from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# 加载数据集
data_dict = {
    'Beach1': [1, 2, 3, 4, 5],
    'Beach2': [6, 7, 8, 9, 10],
    'Beach3': [11, 2, 3, 4, 10],
    'Beach4': [16, 17, 18, 19, 20],
    'Beach5': [11, 12, 13, 5, 1],
    'Beach6': [3, 7, 18, 9, 10],
    'Beach7': [1, 2, 3, 4, 5],
    'Beach8': [2, 2, 3, 4, 50],
    #打乱随机数据

    'Beach1': [2,1,3,5,4],
    'Beach2': [5,2,1,3,8],
    'Beach3': [2,3,4,10,11],
    'Beach4': [10, 14, 12, 14, 12],
    'Beach5': [14, 10, 10, 7, 8],
    'Beach6': [10, 9, 8, 7, 6],
    'Beach7': [1, 2, 3, 4, 5],
    'Beach8': [2, 2, 3, 4, 50],
}
X = []
Y = []

for key in data_dict:
    X.append(data_dict[key])
    Y.append(key)


print(X)
print(Y)
print("-------------------------")
#构造数据集
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)


# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建SVM分类器对象
clf = svm.SVC()

# 在训练集上训练模型
clf.fit(X_train, y_train)
print(f"X是{X_train}\nY是{y_train}")


# 在测试集上进行预测
y_pred = clf.predict(X_test)

#计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)