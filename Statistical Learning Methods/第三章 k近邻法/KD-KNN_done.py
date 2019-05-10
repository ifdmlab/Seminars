# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 23:12:15 2016

@author: zhichen
"""
'''
运用大顶堆优化的KD树来快速搜索最邻近的K个数
完成优化后的KNN算法
使不用再遍历所有数据计算最邻近值
'''
import numpy as np
import operator
from collections import namedtuple  #使元组也可以像class一样命名调用
from pprint import pformat    #这个库的效果是使格式化和print更美观（每行更容易看）
class Node(namedtuple('Node','pointData left right label')): #用Node来表示树的结构
#pointData是自己的值，left为左子树，right为右子树，label为该数据的类别
    def __repr__(self):       #使对于类Node的print会先用pformat进行美观格式化
        return pformat(tuple(self))
    
'''
KD树相关代码
'''
def crateKD(data,labels,depth=0):   #KD树的创建,默认输入的data的最后一列为label
    if not data:
        return None
    den=len(data[0])  #数据的维数    
    axis=depth%int(den)  #根据深度与维数的模来确定二分的轴（可改进为方差最大）
    mid=len(data)//2    #找出中点
    data.sort(key=operator.itemgetter(axis))  #根据二分轴那列数据排序
    return Node(pointData=data[mid],
                left=crateKD(data[:mid],labels[:mid],depth+1),
                right=crateKD(data[mid+1:],labels[mid+1:],depth+1),
                label=labels[mid])
     

#if not dataDateNorm:
 #   print None
#myTree=crateKD(dataDateNorm,uselabel,depth=0)
#myTree=crateKD(dataDateNorm,labelOfMan,depth=0)                
#print myTree

def distance(dataDisA,dataDisB):  #距离计算，欧式
    numberA=np.array([a for a in dataDisA])
    numberB=np.array([b for b in dataDisB])
    return np.sqrt(sum((numberA-numberB)**2))

def search_MaxHeap_KD(target,tree,k):    #大顶堆实现k个最近邻搜索的KDtree查询
    den=len(target)
    depth=0
    axis=depth%den
    treePath=[]     #回溯路径
    while tree:               #顺序查找
        treePath.append(tree)
        if target[axis]>=tree[0][axis]:
            tree=tree[2]
        else:
            tree=tree[1]
        depth+=1
    nearstNode=treePath[-1][0]
    minDis=distance(nearstNode,target)
    
    
    max_heap=[10]*k  #构建大顶堆
    for i in treePath[-1:-k-1:-1]:   
        distanceI=distance(i.pointData,target)
        max_heap_i=[i.pointData,distanceI,i.label]
        max_heap.insert(0,max_heap_i)
    build_max_heap(max_heap,k)
                                  #回溯
    while len(treePath)!=0:      #当回溯路径不为空时，循环
        treeBack=treePath.pop()     #取出路径最后一个点
        depth-=1                    
        axis=depth%den             #该层的分割方向
        if not(treeBack[1] and treeBack[2]):  #如果为叶子，则计算距离
            if distance(treeBack[0],target)<minDis:
                nearstNode=treeBack[0]
                minDis=distance(treeBack[0],target)
                if minDis<max_heap[0][1]:
                    max_heap_plus=[treeBack[0],minDis,treeBack[3]]
                    insert_max_heap(max_heap,max_heap_plus,k)
        else:  #不为叶子，判断该点矩形是否与目标点圆周相交，相交则计算距离
            if abs(treeBack[0][axis]-target[axis])<minDis:
                if distance(treeBack[0],target)<minDis:
                    nearstNode=treeBack[0]
                    minDis=distance(treeBack[0],target)
                    if minDis<max_heap[0][1]:
                        max_heap_plus=[treeBack[0],minDis,treeBack[3]]
                        insert_max_heap(max_heap,max_heap_plus,k)
                if target[axis]<=treeBack[0][axis]:  #进入目标点的反侧查询
                    treeBackPlus=treeBack[2]
                else:
                    treeBackPlus=treeBack[1]
                if treeBackPlus:
                    treePath.append(treeBackPlus)
                    depth+=1
    return max_heap


'''
大顶堆相关函数:遍历到的点对目标点的距离为依据构筑大顶堆，当（新的点，目标点）距离小
于堆顶元素时，替换堆顶元素并确保仍旧维持大顶堆性质
'''
def build_max_heap(data,length):   #构筑大顶堆
    for n in range(length/2,0,-1):
        max_heapify(data,n,length)
    return data
    
def max_heapify(data,i,length):    #维护大顶堆，保证最大值在堆顶
    left=i<<1
    right=left+1
    largest=i
    if left<=length:
        if data[i-1][1]<data[left-1][1]:
            largest=left
    if (right<=length and data[largest-1][1]<data[right-1][1]):
            largest=right
    if largest!=i:
        temp=data[i-1]
        data[i-1]=data[largest-1]
        data[largest-1]=temp
        max_heapify(data,largest,length)
    return data

def extra_max_heap(data,length):   #删除并返回最大值（堆顶），依然保持最大堆
    max_data=data[0]
    data[0]=data[length-1]
    max_heapify(data,1,length-1)
    return max_data
    
def insert_max_heap(data,num,length): #插入一个新的值
    data[0]=num
    max_heapify(data,1,length)
    return data
    
'''
KD-KNN算法     以约会数据为例测试
'''
def classifyKDKnn(target, tree, k):      #KD-knn分类器
    max_heap_knn=search_MaxHeap_KD(target,tree,k)
    classCount={}
    for i in max_heap_knn:
        voteLabel=i[2]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),\
    key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
def file2matrix(filename):          #读取test数据集
    fr=open(filename)
    arrayLine=fr.readlines()
    returnMat=[]
    classLabelVector=[]
    for line in arrayLine:
        line=line.strip()
        lineList=line.split('\t')
        returnMat.append([float(i) for i in lineList[0:3]])
        classLabelVector.append(int(lineList[-1]))
    return returnMat,classLabelVector
    

#dataDate,labelOfMan=file2matrix('datingTestSet2.txt')
#print dataDate,labelOfMan[0:20]    

def autoNorm(dataSet): #数据正则化（由于要计算欧式距离，确保各列数据对距离影响一样）
    dataSetArray=np.array(dataSet)
    minValue=dataSetArray.min(0)
    maxValue=dataSetArray.max(0)
    ranges=maxValue-minValue
    dataSize=dataSetArray.shape[0]
    differenceV=dataSetArray-np.tile(minValue,(dataSize,1))
    dataSetNorm=differenceV/np.tile(ranges,(dataSize,1))
    return dataSetNorm,ranges,minValue
    

#dataDateNorm,ranges,minValue=autoNorm(dataDate)
#print dataDateNorm

def datingClassifyTest(rate):  #测试约会算法准确度
    dataDate,labelOfMan=file2matrix('datingTestSet2.txt')
    dataDateSort=sorted(dataDate,key=operator.itemgetter(0),reverse=True)
    dataDateNorm,ranges,minValue=autoNorm(dataDateSort)
    m=dataDateNorm.shape[0]
    numTest=int(m*rate)
    errorCount=0
    dataDateList=dataDateNorm.tolist()   #将数组转换成列表
    myTree=crateKD(dataDateList[numTest:m],labelOfMan[numTest:m],depth=0)
    for i in range(numTest):
        classifyResult=classifyKDKnn(dataDateNorm[i],myTree,8)
        print ('NO.%d the classify result is:%d,the real answer is:%d '%(\
        i,classifyResult,labelOfMan[i]))
        if (classifyResult!=labelOfMan[i]):errorCount+=1.0
    print ('the error rate is:%f'%(errorCount/float(numTest)))
datingClassifyTest(0.1)
    
def datingClassify():     #约会对象分类算法，自行输入数据
    dataDate,labelOfMan=file2matrix('datingTestSet2.txt')
    dataDateNorm,ranges,minValue=autoNorm(dataDate)
    game=float(raw_input('time to play game:'))
    fly=float(raw_input('airplane miles:'))
    ice=float(raw_input('ice-cream consumed every weak:'))
    dataP=[game,fly,ice]
    answer=classifyKDKnn((dataP-minValue)/ranges,dataDateNorm,labelOfMan,3)
    resultList=['not at all','in small doses','in large doses']
    print ('you will probably like this man:',resultList[answer-1])
#datingClassify()
'''    
dataDate,labelOfMan=file2matrix('C:/study/machine learning code/datingTestSet2.txt')
usedata=dataDate[:10]
uselabel=labelOfMan[:10]
print usedata
print uselabel
dataDateNorm,ranges,minValue=autoNorm(usedata)
print dataDateNorm 
'''