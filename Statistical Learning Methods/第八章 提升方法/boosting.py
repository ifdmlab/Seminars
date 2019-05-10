# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:43:59 2017

@author: Administrator
"""
import numpy as np

def loadSimpData():  
    datMat = np.matrix([  
        [0.0, 1.0, 3.0],[ 0.0, 3.0, 1.0], [ 1.0, 2.0, 2.0], [ 1.0, 1.0, 3.0],    
        [ 1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [ 1.0, 1.0, 2.0], [1.0, 1.0, 1.0],      
        [1.0, 3.0, 1.0], [0.0, 2.0, 1.0]])  
    classLabels = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]  
    return datMat,classLabels  
  
#特征：dimen，分类的阈值是 threshVal,分类对应的大小值是threshIneq  
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data  
    retArray = np.ones((np.shape(dataMatrix)[0],1))  
    if threshIneq == 'lt':  
        retArray[dataMatrix[:,dimen] < threshVal] = -1.0  
    else:  
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0  
    return retArray  
      
#构建一个简单的单层决策树，作为弱分类器  
#D作为每个样本的权重，作为最后计算error的时候多项式乘积的作用  
#三层循环  
#第一层循环，对特征中的每一个特征进行循环，选出单层决策树的划分特征  
#对步长进行循环，选出阈值  
#对大于，小于进行切换  
def buildStump(dataArr,classLabels,D):  
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T  
    m,n = np.shape(dataMatrix)  
    numSteps = 10.0                               #numSteps作为迭代这个单层决策树的步长
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))    
    minError = np.inf                             #init error sum, to +infinity  
    
    for i in range(n):                            #loop over all dimensions  
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()          #第i个特征值的最大最小值  
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(1,int(numSteps)+1):       # 计算阈值，判断最合适的阈值 
            for inequal in ['lt', 'gt']:          # 对大小进行切换
                threshVal = (rangeMin + float(j) * stepSize)  # 计算v值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan  
                errArr = np.mat(np.ones((m,1)))  
                errArr[predictedVals == labelMat] = 0  
                weightedError = D.T*errArr        # 计算错误率  
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)  
                if weightedError < minError:  
                    minError = weightedError  
                    bestClasEst = predictedVals.copy()  
                    bestStump['dim'] = i  
                    bestStump['thresh'] = threshVal  
                    bestStump['ineq'] = inequal  
    return bestStump,minError,bestClasEst  
  
#基于单层决策树的AdaBoost的训练过程  
#numIt 循环次数，表示构造40个单层决策树  
def adaBoostTrainDS(dataArr,classLabels,numIt=40):  
    weakClassArr = []  
    m = np.shape(dataArr)[0]  
    D = np.mat(np.ones((m,1))/m)   #init D to all equal  
    aggClassEst = np.mat(np.zeros((m,1)))  
    for i in range(numIt):  
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump  
        #print "D:",D.T  
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0  
        bestStump['alpha'] = alpha    
        weakClassArr.append(bestStump)                  #store Stump Params in Array  
        #print "classEst: ",classEst.T  
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #exponent for D calc, getting messy  
        D = np.multiply(D,np.exp(expon))                              #Calc New D for next iteration  
        D = D/D.sum()  
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)  
        aggClassEst += alpha*classEst  
        #print "aggClassEst: ",aggClassEst.T  
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))  #这里还用到一个sign函数，主要是将概率可以映射到-1,1的类型  
        errorRate = aggErrors.sum()/m  
        #print "total error: ",errorRate  
        if errorRate == 0.0: break  
    return weakClassArr,aggClassEst    

datamat, classlabels = loadSimpData()
m = np.shape(datamat)[ 0 ]
D = np.mat( np.ones( ( m, 1 ) ) / m )
weakClassArr,aggClassEst = adaBoostTrainDS(datamat,classlabels,40)
print(weakClassArr, aggClassEst)

