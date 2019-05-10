from numpy import *  
import numpy as np  
import pandas as pd  
import math
import operator  
import copy
import re
  
def createDataSet():
    """
    创建数据集
    """
    dataSet = [[1, 1, 'yes' ],
               [1, 1, 'yes' ],
               [1, 2, 'yes' ],
               [0, 0, 'yes' ],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['A', 'B']
    # 返回数据集和每个维度的名称
    return dataSet, labels

# 计算数据集的基尼指数
def calcGini(dataSet):
    numEntries = len(dataSet)
    labelCounts ={}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel]+=1
    Gini =1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        Gini -= prob * prob
    return Gini

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:#若第二个特征为1
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
#计算除了最优特征值以外的基尼指数
def splitOtherDataSetByValue(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] != value:
            reduceFeatVec = featVec[:axis] # 删除这一维特征
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
    :param classList: 类列表
    :retrun: 出现次数最多的类名称
    """

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
#判断特征值是否为两个以上
def binaryZationDataSet(bestFeature,bestSplitValue,dataSet):
        # 求特征标签数
    featList = [example[bestFeature] for example in dataSet]
    uniqueValues = set(featList)

# 特征标签输超过2，对数据集进行二值划分 为了看出决策树构造时的区别，这里特征标签为2时也进行处理
    if len(uniqueValues) >= 2:
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature] == bestSplitValue: # 不做处理
                pass
            else:
                dataSet[i][bestFeature] = 'other'

def chooseBestFeatureToSplitByCART(dataSet):
    numFeatures = len(dataSet[0]) -1
    bestGiniIndex = 1000000.0
    bestSplictValue =[]
    bestFeature = -1
    # 计算Gini指数
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 这里只针对离散变量 & 特征标签
        uniqueVals = set(featList)
        bestGiniCut = 1000000.0
        bestGiniCutValue =[]
        Gini_value =0.0
        # 计算在该特征下每种划分的基尼指数，并且用字典记录当前特征的最佳划分点
        for value in uniqueVals:
            # 计算subDataSet的基尼指数
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            Gini_value = prob * calcGini(subDataSet)
            # 计算otherDataSet的基尼指数
            otherDataSet = splitOtherDataSetByValue(dataSet,i,value)
            prob = len(otherDataSet) / float(len(dataSet))
            Gini_value = Gini_value + prob * calcGini(otherDataSet)
            # 选择最优切分点
            if Gini_value < bestGiniCut:
                bestGiniCut = Gini_value
                bestGiniCutValue = value

        # 选择最优特征向量
        GiniIndex = bestGiniCut
        if GiniIndex < bestGiniIndex:
            bestGiniIndex = GiniIndex
            bestSplictValue = bestGiniCutValue
            bestFeature = i

    # 若当前结点的划分结点特征中的标签超过3个，则将其以之前记录的划分点为界进行二值化处理
    binaryZationDataSet(bestFeature,bestSplictValue,dataSet)
    return bestFeature


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #条件（1）
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #条件（2）
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitByCART(dataSet)
    bestFeatLabel = labels[bestFeat]
    print (bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree




dataSet,labels= createDataSet()
myTree = createTree(dataSet,labels,)
print (myTree)