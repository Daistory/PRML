# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 11:01
# @Author  : ZhenDai
# @Site    : 
# @File    : loadData.py
# @Software: PyCharm

"""
从文件中读取数据
"""


class DataLoader(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_X = []
        self.data_Y = []

    def get_data(self):
        f = open(self.file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.encode('utf-8').decode('utf-8-sig').replace('\n', '')
            line = line.split('\t')
            self.data_X.append(float(line[0]))
            self.data_Y.append(float(line[1]))
        return self.data_X, self.data_Y
