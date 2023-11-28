import numpy as np
import math


import sys


def Dict_to_list(data_dict):
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
        print("I'm here!")
        if X_Temp != []:
            X_average[key]=np.mean(X_Temp)
            X_median[key]=np.median(X_Temp)
        else:
            X_average[key]=0
            X_median[key]=0
        if index_num != []:#有NaN,马上替换

            for index in index_num:
                values[index]=X_average[key] 

        print(f"values:{values}")
        X.append(values)
        Y.append([key]*len(values))
    # X = np.array(X)
    # Y = np.array(Y)
    print("-----------------")
    print(X)
    print(Y)
    print("-----------------")
    print("Yes,Dict_to_list!")
    return X,Y

def Multidimensional_Features_list(data_dict_list):
    X = []
    Y = []
    for dictionary in data_dict_list:
        X_temp,Y_temp=Dict_to_list(dictionary)
        X.append(X_temp)
        Y.append(Y_temp)
    X=np.hstack(X)
    Y=np.hstack(Y)
    Y = np.array([list(set(row)) for row in Y])
    print("*************************")
    print(X)
    print(Y)
    print("*************************")

    return X,Y

NaN=float('nan')
M_dict={'beach1':[1],'beach2':[2],'beach3':[3],'beach4':[NaN],'beach5':[5],'beach6':[6],'beach7':[7],'beach8':[8]}
X_dict={'beach1':[1,2,3,NaN],'beach2':[2,4,5,NaN],'beach3':[3,3,3,3],'beach4':[NaN,2,3,NaN],'beach5':[5,5,5,5],'beach6':[6,6,6,6],'beach7':[6,6,4,7],'beach8':[1,2,3,8]}
Y_dict={'beach1':[NaN],'beach2':[2],'beach3':[3],'beach4':[4],'beach5':[NaN],'beach6':[6],'beach7':[7],'beach8':[8]}
Z_dict={'beach1':[2],'beach2':[2],'beach3':[3],'beach4':[NaN],'beach5':[5],'beach6':[6],'beach7':[7],'beach8':[8]}


X,Y=Multidimensional_Features_list([X_dict,Y_dict,Z_dict])
#print(f"在这{(X,Y)}")
print("___________Final______________")


