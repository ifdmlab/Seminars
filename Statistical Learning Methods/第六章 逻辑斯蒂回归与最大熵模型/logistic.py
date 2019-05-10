#coding:UTF-8  
''''' 
Created on 2015年5月19日 
 
@author: zhaozhiyong 
'''

import sys
from numpy import * 
import matplotlib.pyplot as plt  

def loadDataSet():
    """
    加载数据集
 
    :return:输入向量矩阵和输出向量
    """
    dataMat = []; labelMat = []
    fr = open('./test.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #X0设为1.0，构成拓充后的输入向量
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def plotBestFit(weights):
    """
    画出数据集和逻辑斯谛最佳回归直线
    :param weights:
    """
    dataMat,labelMat=loadDataSet()
    
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #如果weight为0，将第一维的设置的1拿过来进行计算
    if weights is not None:
        x = arange(-3.0, 3.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]   #令w0*x0 + w1*x1 + w2*x2 = 0，其中x0=1，解出x1和x2的关系
        ax.plot(x, y)                               #一个作为X一个作为Y，画出直线
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
# 定义sigmoid函数
# 原函数：f(x)=1/(1+exp(-z))
# 导数： f(x)'=f(x)(1-f(x))
# 因为参数为inX，但使用为-inX,所以为梯度下降算法
def sigmoid(inX):
    return 1/(1+exp(-inX))

"""
# 梯度上升算法
  每个回归系数初始化为1
  重复R次：
      计算整个数据集的梯度
      使用alpha*gradient更新回归系数的向量
      返回回归系数
"""
def gradAscent(dataMatIn, classLabels):
    """
    逻辑斯谛回归梯度上升优化算法
    :param dataMatIn:输入X矩阵（100*3的矩阵，每一行代表一个实例，每列分别是X0 X1 X2）
    :param classLabels: 输出Y矩阵（类别标签组成的向量）
    :return:权值向量
    """
    dataMatrix = mat(dataMatIn)             #转换为 NumPy 矩阵数据类型
    labelMat = mat(classLabels).transpose() #转换为 NumPy 矩阵数据类型 将列表转换为[100.1]的矩阵
    m,n = shape(dataMatrix)                 #矩阵大小    数据为100行3列
    
    alpha = 0.001                           #步长
    maxCycles = 500                         #最大迭代次数，由于计算精度ε需要专业设置
    weights = ones((n,1))

    for k in range(maxCycles):              #最大迭代次数
        h = sigmoid(dataMatrix*weights)     #矩阵内积   计算f(x1)
        error = (labelMat - h)              #向量减法   差值(1-f(x1))
        grad = dataMatrix.transpose() * error #计算梯度  pk=f(x1)(1-f(x1))
        weights += alpha * grad  #矩阵内积  x2=x1+λ*pk
        #判断||f(x2)-f(x1)||<ε  或 ||x2-x1||<ε，停止迭代，令weight=x2
        #否则k=k+1
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升算法
    :param dataMatIn:输入X矩阵（100*3的矩阵，每一行代表一个实例，每列分别是X0 X1 X2）
    :param classLabels: 输出Y矩阵（类别标签组成的向量）
    :param numIter: 迭代次数
    :return:
    """
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)                                           #初始化为单位矩阵
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001                          #步长递减，但是由于常数存在，所以不会变成0
            randIndex = int(random.uniform(0,len(dataIndex)))   #总算是随机了
            h = sigmoid(sum(dataMatrix[randIndex]*weights))     #计算f(x1)
            error = classLabels[randIndex] - h                  #向量减法   差值(1-f(x1))
            weights = weights + alpha * error * dataMatrix[randIndex]   #矩阵内积  x2=x1+λ*pk
            del(dataIndex[randIndex])                           #删除这个样本，以后就不会选到了
    return weights


if __name__ == '__main__':  
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights)








