import os
import pandas as pd
import math

script_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_path, "data.csv")
data = pd.read_csv(file_path) 
print(len(data))
for index in range(len(data)):
    for data_name in ['Water_Temperature','Turbidity']:
        if not math.isnan(data.loc[0][data_name]):
            print("Yes")
