# -*- coding: utf-8 -*-
# @Time    : 2018/9/11 15:57
# @Author  : ZhenDai
# @Site    : 
# @File    : logistic_bayes_model.py
# @Software: PyCharm
import numpy as np


def sigmoid(param):
    return 1.0 / (1 + np.exp(-param))


class LogisticBayesClassification(object):
    def __init__(self, x_train, y_train, x_test, y_test, weight_num=10):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._weight_num = weight_num
        self.weight_matrix = np.mat(np.ones((self._weight_num * self._x_train.shape[0], 1)))
        self._design_matrix = np.zeros((self._weight_num * self._x_train.shape[0], 1))
        self._S0 = np.eye(self._weight_num * self._x_train.shape[0])
        self._M0 = np.mat(np.ones((self._weight_num * self._x_train.shape[0], 1)))
        self._SN = np.zeros((self._weight_num * self._x_train.shape[0], self._weight_num * self._x_train.shape[0]))

    def _gen_cov_matrix(self):
        for i in range(self._weight_num * self._x_train.shape[0]):
            for j in range(i + 1):
                if i == j:
                    self._S0[i][i] = np.random.normal(20, 5, 1)[0]
                else:
                    self._S0[i][j] = self._S0[j][i] = np.random.normal(5, 2, 1)[0]

    def _get_SN(self):
        self._SN = np.mat(self._S0).I
        for i in range(self._x_train.shape[1]):
            design_temp = np.mat(self._basis_trans(self._x_train[:, i:i + 1]))
            y = float(sigmoid(self.weight_matrix.T * design_temp))
            self._SN += y * (1 - y) * design_temp * design_temp.T
        self._SN = np.mat(self._SN).I

    def _basis_trans(self, x):
        for i in range(self._weight_num):
            for j in range(x.shape[0]):
                self._design_matrix[i * x.shape[0] + j][0] = np.power(x[j][0], i)
        return self._design_matrix

    def _grad_ascent(self):
        max_cycles = 10000
        alpha = 0.01
        for i in range(max_cycles):
            error = np.mat(np.zeros((self._weight_num * self._x_train.shape[0], 1)))
            for j in range(self._x_train.shape[1]):
                design_temp = self._basis_trans(self._x_train[:, j:j + 1])
                error = error + float(
                    sigmoid(self.weight_matrix.T * design_temp) - float(self._y_train[j])) * design_temp
            self.weight_matrix = self.weight_matrix - alpha * np.mat(
                error + np.mat(self._S0).I * self.weight_matrix - np.mat(self._S0).I * self._M0)

    def get_weight(self):
        self._grad_ascent()
        return self.weight_matrix

    def save_model(self):
        f = open('logistic_Bayes_mode.json', 'w')
        for i in range(self.weight_matrix.shape[0]):
            f.write(str(self.weight_matrix.getA()[i][0]))
            f.write('\n')
        f.close()

    def load_model(self):
        self.weight_matrix = np.loadtxt('logistic_Bayes_mode.json').reshape(-1, 1)

    def predict(self, data):
        design_temp = np.mat(self._basis_trans(data))
        mean = self.weight_matrix.T * design_temp
        var = design_temp.T * self._SN * design_temp
        value = sigmoid(np.power(1 + np.math.pi * var / 8, -0.5) * mean)
        if value >= 0.5:
            return 1
        else:
            return 0

    def get_test_accuracy(self):
        index = 0
        for i in range(self._x_test.shape[1]):
            y_temp = self.predict(self._x_test[:, i:i + 1])
            if abs(y_temp - float(self._y_test[i])) <= 0.5:
                index += 1
        acc = float(index) / float(self._y_test.shape[0])
        return acc
