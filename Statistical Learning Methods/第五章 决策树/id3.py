import math
import operator

#计算经验熵
def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    #计算每种结果的个数，yes2个，no3个
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*math.log(prob, 2)
    return shannonEnt
     
def CreateDataSet():
    #两个特征，分别为1,0
    dataset = [[1, 1, 'yes' ],
               [1, 1, 'yes' ],
               [1, 2, 'yes' ],
               [0, 0, 'yes' ],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['A', 'B']
    return dataset, labels
#找出所有第axis个特征为value的 
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:#若第二个特征为1
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
#Cd3
def chooseBestFeatureToSplitCd3(dataSet):
    numberFeatures = len(dataSet[0])-1#特征的数量
    baseEntropy = calcShannonEnt(dataSet)#计算经验熵
    bestInfoGain = 0.0;
    bestFeature = -1;
    for i in range(numberFeatures):
        featList = [example[i] for example in dataSet]#特征的所有值
        uniqueVals = set(featList)#特征的集合
        newEntropy =0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #信息增益=经验熵-经验条件熵
        infoGain = baseEntropy - newEntropy
        #选择信息增益最大的
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#D4.5
def chooseBestFeatureToSplitD45(dataSet):  
    numFeatures = len(dataSet[0])-1  #求属性的个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1  
    for i in range(numFeatures):  #求所有属性的信息增益
        featList = [example[i] for example in dataSet]  
        uniqueVals = set(featList)  #第i列属性的取值（不同值）数集合
        newEntropy = 0.0  
        splitInfo = 0.0;
        for value in uniqueVals:  #求第i列属性每个不同值的熵*他们的概率
            subDataSet = splitDataSet(dataSet, i , value)  
            prob = len(subDataSet)/float(len(dataSet))  #求出该值在i列属性中的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  #求i列属性各值对于的熵求和
            splitInfo -= prob * math.log(prob, 2);
        infoGain = (baseEntropy - newEntropy) / splitInfo;  #求出第i列属性的信息增益率
        if(infoGain > bestInfoGain):  #保存信息增益率最大的信息增益率值以及所在的下表（列值i）
            bestInfoGain = infoGain
            bestFeature = i  
    return bestFeature  

#选择实例数最多的
def majorityCnt(classList):
    classCount ={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]=1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
  
 
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #条件（1）
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #条件（2）
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    #####################################
    bestFeat = chooseBestFeatureToSplitCd3(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        print (myTree)
    return myTree
 
         
         
myDat,labels = CreateDataSet()
myTree = createTree(myDat, labels)