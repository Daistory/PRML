# -*- coding: utf-8 -*-
# @Time    : 2018/9/3 9:09
# @Author  : ZhenDai
# @Site    :
# @File    : logistic_mian.py
# @Software: PyCharm
"""
logistic回归的运行脚本
"""
import loadData
import logistic_bayes_model
import numpy as np


def trans_label(label_matrix):
    temp_list = []
    for label in label_matrix.T:
        if label.getA()[0][1] == 1.0:
            temp_list.append(0)
        else:
            temp_list.append(1)
    return temp_list


if __name__ == "__main__":
    file_path = r'iris.txt'
    x_train, x_test, y_train, y_test = loadData.DataLoader(file_path).get_train_test_data()
    y_train_list = trans_label(y_train)
    y_test_list = trans_label(y_test)
    #
    logistic_model = logistic_bayes_model.LogisticBayesClassification(x_train, np.mat(y_train_list).T, x_test,
                                                                      np.mat(y_test_list).T)
    weight = logistic_model.get_weight()
    # logistic_model.load_model()
    logistic_model.save_model()
    for i in range(x_test.shape[1]):
        print(logistic_model.predict(x_test[:, i:i + 1]))
        print(y_test_list[i])
        print('\n')
    print("accuracy:")
    print(logistic_model.get_test_accuracy())
