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

#生成决策树
def createTree(dataSet,labels,data_full,labels_full):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet,labels)
    
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    
    featureVals = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureVals) 
    currentLabel = labels_full.index(labels[bestFeature])
    featureVals_full = [example[currentLabel] for example in data_full]
    uniqueVals_full = set(featureVals_full)
    
    del(labels[bestFeature])
    for value in uniqueVals:
        sublabels = labels[:]
        
        uniqueVals_full.remove(value)
        myTree[bestFeatureLabel][value] = createTree(splitDataSet \
            (dataSet,bestFeature,value), sublabels,data_full,labels_full)
    
    for val in uniqueVals_full:
        myTree[bestFeatureLabel][val] = majorityCnt(classList)
    return myTree

#main function
df = pd.read_csv('WaterMelon_4_4.txt',sep='\t')
data_full = df.values[:10,1:].tolist()
dataSet = data_full[:]
labels_full = df.columns[1:-1].tolist()
labels = labels_full[:]
myTree = createTree(dataSet,labels,data_full,labels_full)

createPlot(myTree)
#print(myTree,end='\n')
#print(labels_full)