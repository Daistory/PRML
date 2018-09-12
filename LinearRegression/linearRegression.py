# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 11:27
# @Author  : ZhenDai
# @Site    : 
# @File    : linearRegression.py
# @Software: PyCharm
"""
线性回归模型
"""
import numpy as np


class LinearRegression(object):
    def __init__(self, train_data_x, train_data_y, alpha=1, beta=1, weight_num=10):
        self.train_data_X = np.array(train_data_x).reshape(-1, 1)
        self.train_data_Y = np.array(train_data_y).reshape(-1, 1)
        self.alpha = alpha
        self.beta = beta
        self.weight_num = weight_num
        self.mn = np.zeros((self.weight_num, 1))
        self.sn = np.zeros((self.weight_num, self.weight_num))

    def _get_design_matrix(self):
        size = len(self.train_data_X)
        design_matrix = np.ones((size, self.weight_num))
        for i in range(size):
            for j in range(self.weight_num):
                design_matrix[i][j] = pow(self.train_data_X[i], j)
        return np.mat(design_matrix)

    def _get_hyper_para(self):
        design_matrix = self._get_design_matrix()
        eye_matrix = np.eye(self.weight_num)
        while True:
            prior_beta = self.beta
            prior_alpha = self.alpha
            eig_val, eig_vec = np.linalg.eig(self.beta * design_matrix.T * design_matrix)
            r = 0.0
            for i in eig_val:
                if str(type(i)) != '<class \'numpy.float64\'>':
                    print(i)
                    continue
                r += i / (i + self.alpha)
            sn = self.alpha * eye_matrix + self.beta * design_matrix.T * design_matrix
            mn = self.beta * sn.I * design_matrix.T * self.train_data_Y
            self.alpha = float(r / (mn.T * mn))
            loss = 0.0
            for i in range(len(self.train_data_X)):
                loss += pow((self.train_data_Y[i] - mn.T * (design_matrix[i]).T), 2)
            self.beta = float((len(self.train_data_X) - r) / loss)
            print(self.alpha)
            print('\n')
            print('\n')
            print(self.beta)
            print('.......................\n')
            if abs(prior_alpha - self.alpha) <= 0.001 and abs(prior_beta - self.beta) <= 0.001:
                break

    def _get_para(self):
        self._get_hyper_para()
        design_matrix = self._get_design_matrix()
        size = design_matrix.shape[1]
        self.sn = (self.alpha * np.eye(size) + self.beta * design_matrix.T * design_matrix).I
        self.mn = self.beta * self.sn * design_matrix.T * self.train_data_Y

    def train_model(self):
        self._get_para()

    def predict_model(self, x):
        base = np.zeros(self.mn.shape)
        for i in range(self.mn.shape[0]):
            base[i][0] = pow(x, i)
        mean = self.mn.T * base
        var = 1 / self.beta + base.T * self.sn * base
        return float(np.random.normal(mean, var))

    def model_save(self, file_path):
        f = open(file_path, 'w')
        f.write(str(self.alpha) + '\t')
        f.write(str(self.beta))
        f.close()

    def model_load(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.encode('utf-8').decode('utf-8-sig').replace('\n', '')
            line = line.split('\t')
            self.alpha = (float(line[0]))
            self.beta = (float(line[1]))
        self._get_para()
