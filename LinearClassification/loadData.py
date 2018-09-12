# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 14:45
# @Author  : ZhenDai
# @Site    : 
# @File    : loadData.py
# @Software: PyCharm

"""
从文件中读取数据，划分成数据和标签
"""
import numpy as np
from sklearn import model_selection


class DataLoader(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def _get_data(self):
        data = np.loadtxt(self.file_path, delimiter=',')
        return data[:, 0:-1], data[:, -1]

    def normalized(self, data):
        shape_size = data.shape
        for i in range(shape_size[1]):
            max_num = max(data[:, i:i + 1])
            min_num = min(data[:, i:i + 1])
            for j in range(shape_size[0]):
                data[j][i] = float(data[j][i] - min_num) / (max_num - min_num)
        return data

    def get_train_test_data(self):
        temp_data, temp_label = self._get_data()
        temp_data = self.normalized(temp_data)
        label = np.zeros((len(temp_label), 3))
        for i in range(len(temp_label)):
            label[i][int(temp_label[i]) - 1] = 1

        x_train, x_test, y_train, y_test = model_selection.train_test_split(temp_data, label, test_size=0.3,
                                                                            random_state=0)
        return np.mat(x_train).T, np.mat(x_test).T, np.mat(y_train).T, np.mat(y_test).T
