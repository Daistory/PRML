# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 21:09
# @Author  : ZhenDai
# @Site    :
# @File    : main.py
# @Software: PyCharm

import loadData
import fisherModel
import numpy as np
from matplotlib import pyplot as plt

"""
fisher模型的运行脚本
"""
if __name__ == "__main__":
    file_path = r'iris.data.txt'
    x_train, x_test, y_train, y_test = loadData.DataLoader(file_path).get_train_test_data()

    W = fisherModel.FisherClassification(x_train, y_train, x_test, y_test).get_para()
    # print(x_train)

    L = [[], [], []]
    for i in range(x_train.shape[1]):
        index = int(np.argwhere(y_train[:, i] > 0)[0][0])
        L[index].append(float(W.T * np.mat(x_train[:, i])))
    print(L)
    index = 0
    for i in L:
        index += 1
        plt.scatter(i, [j * 0 for j in range(len(i))], label=str(index), s=30)
    plt.legend(loc='upper right')
    plt.show()
