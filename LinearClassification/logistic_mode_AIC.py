# -*- coding: utf-8 -*-
# @Time    : 2018/8/31 16:12
# @Author  : ZhenDai
# @Site    : 
# @File    : logistic_mode_AIC.py
# @Software: PyCharm
import numpy as np


def sigmoid(param):
    return 1.0 / (1 + np.exp(-param))


class LogisticAICClassification(object):
    def __init__(self, x_train, y_train, x_test, y_test, weight_num=10):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._weight_num = weight_num
        self.weight_matrix = np.mat(np.ones((self._weight_num * self._x_train.shape[0], 1)))
        self._design_matrix = np.zeros((self._weight_num * self._x_train.shape[0], self._x_train.shape[1]))

    def _basis_trans(self, data_matrix):
        for i in range(self._weight_num):
            for j in range(data_matrix.shape[0]):
                for k in range(data_matrix.shape[1]):
                    self._design_matrix[i * data_matrix.shape[0] + j][k] = np.power(self._x_train.getA()[j][k], i)
        self._design_matrix = np.mat(self._design_matrix).T
        return self._design_matrix[0:data_matrix.shape[1], ]

    def _grad_ascent(self):
        self._basis_trans(self._x_train)
        max_cycles = 10000
        alpha = 0.01
        for i in range(max_cycles):
            temp_y = np.mat(sigmoid(self._design_matrix * self.weight_matrix))
            self.weight_matrix = self.weight_matrix - alpha * np.mat(self._design_matrix).T * (self._y_train - temp_y)
        # 使用N-R计算，在求矩阵的逆的时候，出现奇异矩阵
        # label_size = self._y_train.shape[0]
        # R = np.eye(label_size)
        # for i in range(max_cycles):
        #     temp_y = sigmoid(self._design_matrix * self._weight_matrix).getA()
        #     for k in range(label_size):
        #         R[k][k] = temp_y[k][0] * (1 - temp_y[k][0])
        #     self._weight_matrix = self._weight_matrix - (
        #             np.mat(self._design_matrix).T * np.mat(R) * np.mat(self._design_matrix)).I * np.mat(
        #         self._design_matrix).T * (np.mat(temp_y)-self._y_train)

    def get_weight(self):
        self._grad_ascent()
        return self.weight_matrix

    def save_model(self):
        f = open('logistic_AIC_mode.json', 'w')
        for i in range(self.weight_matrix.shape[0]):
            f.write(str(self.weight_matrix.getA()[i][0]))
            f.write('\n')
        f.close()

    def load_model(self):
        self.weight_matrix = np.loadtxt('logistic_AIC_mode.json').reshape(-1, 1)

    def predict(self, data):
        self._basis_trans(data)
        if sigmoid(self._basis_trans(data) * self.weight_matrix) >= 0.5:
            return 1
        else:
            return 0

    def get_test_accuracy(self):
        y_temp = sigmoid(self._basis_trans(self._x_test) * self.weight_matrix)
        print(y_temp)
        error = np.mat(y_temp) - self._y_test
        print(error)
        index = 0
        for i in error.getA():
            if abs(i) <= 0.5:
                index += 1
        print(index)
        acc = float(index) / float(self._y_test.shape[0])
        return acc
