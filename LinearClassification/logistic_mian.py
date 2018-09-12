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

if __name__ == "__main__":
    file_path = r'iris.txt'
    x_train, x_test, y_train, y_test = loadData.DataLoader(file_path).get_train_test_data()
    y_train_list = []
    y_test_list = []
    print(y_train.T)
    for i in y_train:
        print(i)
        exit(0)
