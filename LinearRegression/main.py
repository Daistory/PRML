# -*- coding: utf-8 -*-
# @Time    : 2018/8/11 14:31
# @Author  : ZhenDai
# @Site    : 
# @File    : main.py
# @Software: PyCharm

import loadData
import generateData
import linearRegression


def get_RMSE(data_Y, pre_Y):
    m = len(data_Y)
    loss = 0.0
    for i in range(m):
        loss += pow((data_Y[i] - pre_Y[i]), 2)
    return pow(loss / m, 0.5)


if __name__ == "__main__":
    file_path = 'data.txt'
    # 从文件中加载数据
    X, Y = loadData.DataLoader(file_path).get_data()

    # 划分训练集测试集
    train_data_X, train_data_Y, test_data_X, test_data_Y = generateData.DataGenerator(X, Y).rand_divide()

    # 训练模型
    liner_model = linearRegression.LinearRegression(train_data_X, train_data_Y)
    liner_model.train_model()

    # 预测
    pre_Y = []
    f1 = open('pre.txt', 'w')
    for i in test_data_X:
        pre = liner_model.predict_model(i)
        f1.write(str(pre) + '\n')
        pre_Y.append(pre)

    # 计算rmse
    rmse = get_RMSE(test_data_Y, pre_Y)
    print(rmse)
    f = open('rmse.txt', 'w')
    f.write(str(rmse))
    f.close()
    f1.close()

    # 保存模型
    liner_model.model_save('linear.model')

    """
    加载已经保存的模型
    linear_model = linearRegression.LinearRegression(train_data_X, train_data_Y)
    linear_model.model_load('linear.model')
    # 进行预测
    linear_model.predict_model()
    """
