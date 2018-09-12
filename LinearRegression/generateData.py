# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 10:08
# @Author  : ZhenDai
# @Site    : 
# @File    : generateData.py
# @Software: PyCharm
import numpy as np

"""
用来划分训练集测试集
"""


class DataGenerator(object):
    def __init__(self, data_x, data_y):
        self.data_X = data_x
        self.data_Y = data_y
        self.train_data_X = []
        self.train_data_Y = []
        self.test_data_X = []
        self.test_data_Y = []

    def rand_divide(self, test_ratio=0.3):
        size = len(self.data_X)
        for i in range(int(test_ratio * size)):
            index = np.random.randint(0, len(self.data_Y)-1)
            self.test_data_X.append(self.data_X[index])
            self.test_data_Y.append(self.data_Y[index])
            del self.data_X[index]
            del self.data_Y[index]
        self.train_data_X = self.data_X
        self.train_data_Y = self.data_Y
        return self.train_data_X, self.train_data_Y, self.test_data_X, self.test_data_Y
