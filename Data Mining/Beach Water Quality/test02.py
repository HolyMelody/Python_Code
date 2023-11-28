#画佛祖保佑函数
def buddha_bless():
    print("　　　　　　　　┏┓　　　┏┓")
    print("　　　　　　　┏┛┻━━━┛┻┓")
    print("　　　　　　　┃　　　　　　　┃")
    print("　　　　　　　┃　　　━　　　┃")
    print("　　　　　　　┃　┳┛　┗┳　┃")
    print("　　　　　　　┃　　　　　　　┃")
    print("　　　　　　　┃　　　┻　　　┃")
    print("　　　　　　　┃　　　　　　　┃")
    print("　　　　　　　┗━┓　　　┏━┛")
    print("　　　　　　　　　┃　　　┃")
    print("　　　　　　　　　┃　　　┃")
    print("　　　　　　　　　┃　　　┗━━━┓")
    print("　　　　　　　　　┃　　　　　　　┣┓")
    print("　　　　　　　　　┃　　　　　　　┏┛")
    print("　　　　　　　　　┗┓┓┏━┳┓┏┛")
    print("　　　　　　　　　　┃┫┫　┃┫┫")
    print("　　　　　　　　　　┗┻┛　┗┻┛")

# 调用函数，输出佛祖保佑的图案
def fo():
    # 构建图形的每一行
    lines = [
        "  *                             _ooOoo_",
        "  *                            o8888888o",
        "  *                            88\" . \"88",
        "  *                            (| -_- |)",
        "  *                            O\\  =  /O",
        "  *                         ____/`---'\\____",
        "  *                       .'  \\\\|     |//  `.",
        "  *                      /  \\\\|||  :  |||//  \\",
        "  *                     /  _||||| -:- |||||-  \\",
        "  *                     |   | \\\\\\  -  /// |   |",
        "  *                     | \\_|  ''\\---/''  |   |",
        "  *                     \\  .-\\__  `-`  ___/-. /",
        "  *                   ___`. .'  /--.--\\  `. . __",
        "  *                .\"\" '<  `.___\\_<|>_/___.'  >'\"\".",
        "  *               | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |",
        "  *               \\  \\ `-.   \\_ __\\ /__ _/   .-`  /  /",
        "  *          ======`-.____`-.___\\_____/___.-`____.-'======",
        "  *                             `=---='",
        "  *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        "  *                     佛祖保佑        永无BUG",
        "  *            佛曰:",
        "  *                   写字楼里写字间，写字间里程序员；",
        "  *                   程序人员写程序，又拿程序换酒钱。",
        "  *                   酒醒只在网上坐，酒醉还来网下眠；",
        "  *                   酒醉酒醒日复日，网上网下年复年。",
        "  *                   但愿老死电脑间，不愿鞠躬老板前；",
        "  *                   奔驰宝马贵者趣，公交自行程序员。",
        "  *                   别人笑我忒疯癫，我笑自己命太贱；",
        "  *                   不见满街漂亮妹，哪个归得程序员？"
    ]

    # 打印图形
    for line in lines:
        print(line)

dict1 = {
    'Beach1': 10,
    'Beach2': 20,
    'Beach3': 30,
}

dict2 = {
    'Beach1': 11,
    'Beach2': 21,
    'Beach3': 33,
    'Beach4': 45,
    'Beach8': 80,
}

dict3 = {
    'Beach1': 13,
    'Beach2': 25,
    'Beach3': 33,
    'Beach8': 89,
    'Beach9': 90,
}

# 创建空列表存储结果
X = []

Y = []

# 将所有字典合并为一个列表
dict_list = [dict1, dict2, dict3]

# 遍历字典列表
i=0
j=0
for dictionary in dict_list:
    # 遍历字典中的键值对
            for key in data_dict:
            Remove_NaN = [[x] for x in data_dict[key] if not math.isnan(x)]
            X_average[key]=np.mean(Remove_NaN)
            X_median[key]=np.median(Remove_NaN)
            X=X+Remove_NaN
            Y=Y+[[key]]*len(Remove_NaN)
    for key, value in dictionary.items():
        # 查找键在X中的索引
        i=i+1
        print(f"-----------{i}-------------")
        print(f"key是{key},value是{value}")
        index = None
        for i, sublist in enumerate(X):#得到索引和值
            j=j+1
            print(f"_________&_{j}_&___________")
            print(f"i是{i},sublist是{sublist}")
            if key in sublist:
                index = i
                break
            print("__________*__{j}_*________")
        # 如果键已存在于X中，则将值添加到对应的子列表中
        print(f"{[index]*10}")
        if index is not None:
            X[index].append(value)
            print(f"len(X)是{len(X)}len(Y)是{len(Y)}")
            #画佛祖保佑
            fo()
        # 如果键不存在于X中，则创建新的子列表并添加键和值
        else:
            X.append([value])
            Y.append([key])
            buddha_bless()

        print(f"------------{i}------------")

print(len(X)==len(Y))
print("X:", X)
print("Y:", Y)