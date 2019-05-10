# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 15:34:20 2018

@author: Administrator
"""
import numpy as np

def loadDaaSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# i是alpha的下标， m是所有alpha的数目
def selectJrand(i, m):
    j=i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

# 用于调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj >H:
        aj = H
    if L> aj:
        aj = L
    return aj



