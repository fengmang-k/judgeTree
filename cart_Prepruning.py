import numpy as np 
import pandas as pd 
import operator
from PlotTree import createPlot
#计算数据集dataSet的基尼值
def gini_Value(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    giniValue = 1
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        giniValue-=prob**2
    return giniValue     
#求离散属性取某一值时数据集
def splitDataSet(dataSet,axis,value):
    subDataSet=[]
    for example in dataSet:
        reducedExample = []
        if example[axis] == value:
            reducedExample = example[:axis]
            reducedExample.extend(example[axis+1:])
            subDataSet.append(reducedExample)
    return subDataSet
#求最佳划分属性
bestSplitDict={}
def chooseBestFeatureToSplit(dataSet,labels):
    
    miniGini_Index = 10000
    bestFeature = -1
    
    for i in range(len(labels)):
        featureList = [example[i] for example in dataSet]
        uniqueValue = set(featureList)
        gini_Index = 0.0
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet,i,value)
            prob= len(subDataSet)/float(len(dataSet))
            gini_Index += prob*gini_Value(subDataSet)
        #
        if gini_Index < miniGini_Index:
            bestFeature = i
            miniGini_Index = gini_Index

    return bestFeature
#当不能划分时，投票选出分类
def majorityCnt(classList):
    classCount={}
    for i in classList:
        if i not in classCount.keys():
            classCount[i]=0
        classCount[i]+=1
    return max(classCount,key=classCount.get)
#节点对验证集数据正确划分数量
def rightSplitNum(dataSet,testData):
    dataClass = {}
    for example in dataSet:
        if example[-1] not in dataClass.keys():
            dataClass[example[-1]] = 0
        dataClass[example[-1]] += 1
    splitClass = max(dataClass,key=dataClass.get)
    rightNum = 0
    for example in testData:
        if example[-1] == splitClass:
            rightNum += 1
    return rightNum
#生成决策树
def createTree(dataSet,labels,data_full,labels_full,testData):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet,labels)
    ##确定当前节点是否继续划分（预剪枝）
    preSplitRightNum = rightSplitNum(dataSet,testData)
    featureVals = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureVals)
    currentLabel = labels_full.index(labels[bestFeature])
    featureVals_full = [example[currentLabel] for example in data_full]
    uniqueVals_full = set(featureVals_full)
    postSplitRightNum = 0
    for value in uniqueVals:
        uniqueVals_full.remove(value)
        subDataSet = splitDataSet (dataSet,bestFeature,value)
        subTestData = splitDataSet (testData,bestFeature,value)
        postSplitRightNum += rightSplitNum(subDataSet,subTestData)
    for value in uniqueVals_full:
        subTestData = splitDataSet (testData,bestFeature,value)
        postSplitRightNum += rightSplitNum(dataSet,subTestData)
    ##无需划分，返回根节点
    if preSplitRightNum >= postSplitRightNum:
        return majorityCnt(classList)
    ##可以划分，进行划分
    else:
        bestFeatureLabel = labels[bestFeature]
        myTree = {bestFeatureLabel:{}}
        del(labels[bestFeature])
        subUniqueVals_full = set(featureVals_full)
        for value in uniqueVals:
            sublabels = labels[:]
            subUniqueVals_full.remove(value)
            myTree[bestFeatureLabel][value] = createTree(splitDataSet \
                (dataSet,bestFeature,value), sublabels,data_full,labels_full, \
                    splitDataSet(testData,bestFeature,value))
        for value in subUniqueVals_full:
            myTree[bestFeatureLabel][value] = majorityCnt(classList)
        return myTree
#main function
df = pd.read_csv('WaterMelon_4_4.txt',sep='\t')
data_full = df.values[:,1:].tolist()
dataSet = data_full[:11]
testData = data_full[11:]
labels_full = df.columns[1:-1].tolist()
labels = labels_full[:]
myTree = createTree(dataSet,labels,data_full,labels_full,testData)
createPlot(myTree)
#print(myTree,end='\n')
#print(labels_full)