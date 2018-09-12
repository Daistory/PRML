# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 16:02
# @Author  : ZhenDai
# @Site    : 
# @File    : fisherModel.py
# @Software: PyCharm

import numpy as np


class FisherClassification(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.weight_matrix = np.ones((x_train.shape[0], 1))

    def _get_mean(self):
        # mean_list为各类的均值其中index = -1表示的是global mean
        mean_list = []
        class_num = []
        for i in range(self.y_train.shape[0] + 1):
            mean_list.append(np.zeros((self.x_train.shape[0], 1)))
            class_num.append(int(0))
        for i in range(self.x_train.shape[1]):
            mean_list[-1] += self.x_train[:, i]
            class_num[-1] += 1
            # print(self.y_train[:, i])
            index = int(np.argwhere(self.y_train[:, i] > 0)[0][0])
            mean_list[index] += self.x_train[:, i]
            class_num[index] += 1
        for i in range(len(mean_list)):
            mean_list[i] = mean_list[i] / float(class_num[i])
        return mean_list, class_num

    def get_para(self):
        mean_list, class_num = self._get_mean()
        sb = np.zeros((len(mean_list[0]), len(mean_list[0])))
        sw = np.zeros((len(mean_list[0]), len(mean_list[0])))
        for i in range(len(mean_list) - 1):
            sb = sb + class_num[i] * (mean_list[i] - mean_list[-1]) * (mean_list[i] - mean_list[-1]).T
        for i in range(self.x_train.shape[1]):
            index = int(np.argwhere(self.y_train[:, i] > 0)[0][0])
            sw += (self.x_train[:, i] - mean_list[index]) * (self.x_train[:, i] - mean_list[index]).T
        eig_val, eig_vec = np.linalg.eig(np.mat(sw).I * np.mat(sb))
        self.weight_matrix = eig_vec[:, int(np.argwhere(eig_val == max(eig_val)))]
        return self.weight_matrix
