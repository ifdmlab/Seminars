# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import math
import numpy as np
from collections import Counter

def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):#创建词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建并集
    #print(vocabSet)        
    return list(vocabSet)

#词袋模型
def bagOfWord2VecMN(vocabList,inputSet):#根据词汇表，讲句子转化为向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  #针对一个词在文档中出现不止一次情况
    #print("returnVec + ", returnVec)
    return returnVec

########训练算法（朴素贝叶斯分类器训练函数(此处仅处理两类分类问题)）########
#计算每个类别中的文档数目
#对每篇训练文档：
    #对每个类别：
        #如果词条出现文档中->增加该词条的计数值
        #增加所有词条的计数值
    #对每个类别：
        #对每个词条：
            #将该词条的数目除以总词条数目得到条件概率
    #返回每个类别的条件概率
    
# 公式：P(Ci|w)=P(w|Ci)P(Ci)/P(w)  某词向量X为分类Ci的概率
#  p(w|ci) = p(w0|ci) * p(w1|ci) * p(w2|ci) * .... * p(wn|ci) wn指的是词向量中的某个单词
###############    
def trainNB0(trainMatrix,trainCategory):
    '''
     trainMatrix:文档矩阵
     trainCategory:每篇文档类别标签
    '''
    numTrainDocs = len(trainMatrix)  #测试数据数量
    numWords = len(trainMatrix[0])   #单行数据数量
    pAbusive = (sum(trainCategory)+1)/(float(numTrainDocs)+sum(trainCategory))  #先验概率
    print("pAbusive:",pAbusive,"\n")
    
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)      # 计算频数初始化为1
    p0Denom = 2.0                  # Sj*λ
    p1Denom = 2.0                  #初始化分母为2，避免出现概率为0，即拉普拉斯平滑
    
    for i in range(numTrainDocs):
        if trainCategory[i]==1:   # 判断如果是侮辱性文字
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            #print("p1Num:",p1Num, "\n","p1Denom:",p1Denom)
        else:                     # 判断如果是正常言论
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            #print("p0Num:",p0Num, "\n", "p0Denom:",p0Denom)
            
    print("p0Num:",p0Num, "\n", "p1Num:",p1Num, "\n", "p0Denom:",p0Denom)
    #p1Vect = math.log(p1Num/p1Denom)#注意  侮辱性文字条件概率
    p1Vect = p1Num/p1Denom  #应该将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p0Vect = p0Num/p0Denom  #注意  正常言论条件概率
    print("p0Vect:",p0Vect, "p1Vect:",p1Vect)
    return p0Vect,p1Vect,pAbusive#返回各类对应特征的条件概率向量
                                 #和各类的先验概率
                                 
############分类函数#############
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    """
    vec2Classify:要分类的向量
    p0Vec,p1Vec,pClass1：分别对应trainNBO计算得到的3个概率
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)#注意
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)#注意
    print("p0:",p0,"p1:",p1)
    if p1 > p0:
        return 1
    else:
        return 0

def main():
    listOPosts,listClasses = loadDataSet()#加载数据
    myVocabList = createVocabList(listOPosts)#建立词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bagOfWord2VecMN(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)#训练
    testEntry = ['love','my','dalmation']
    thisDoc = bagOfWord2VecMN(myVocabList,testEntry)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    
if __name__ == '__main__':
    main()