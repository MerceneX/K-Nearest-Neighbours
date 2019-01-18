import math, operator, matplotlib, sys, getopt
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing

def cmdParameters():
    trainCSVName, testCSVName, userSpecifiedK, confMatrix, result, argv = '','' , 5, False, False, sys.argv[1:]
    opts, agrs = getopt.getopt(argv, "t:T:irk:")
    for opt,arg in opts:
        if opt == '-t': trainCSVName = arg
        if opt == '-T': testCSVName = arg
        if opt == '-k': userSpecifiedK = int(arg)
        if opt == '-i': confMatrix = True
        if opt == '-r': result = True
    return trainCSVName, testCSVName, userSpecifiedK, confMatrix, result

def euclidDistance(train, test):
	distance = 0
	for i in range(len(train)-1): distance += pow((train[i] - test[i]), 2)
	return math.sqrt(distance)

def sortNeighbors(unsortedList,k):
    neighborsSorted, sortedNeighbours = sorted(unsortedList.items(), key=operator.itemgetter(1)), list()
    for i in range(k): sortedNeighbours.append(neighborsSorted[i])
    return sortedNeighbours

def getClosestNeighbours(trainSet,testData,k):
    feature, distances = list(), dict()
    for i in range(len(trainSet)):
        for columnName in trainSet: feature.append(trainSet[columnName][i])
        distance = euclidDistance(feature, testData)
        distances[i] = distance
        feature.clear()
    return sortNeighbors(distances, k)

def classify(neighbors,trainSet):
    neighborsClasses = list()
    for i in range(len(neighbors)):
        neighborIndex = neighbors[i][0]
        neighborsClasses.append(trainSet[trainSet.columns[-1]][neighborIndex])
    prediction = max(set(neighborsClasses),key=neighborsClasses.count)
    return prediction

def iterate(trainSet, testSet, classNames, k):
    classified = list()
    for i in range(len(testSet)):
        row = list()
        for columnName in testSet:
            row.append(testSet[columnName][i])
        neighbors = getClosestNeighbours(trainSet,row, k)
        classified.append(classify(neighbors,trainSet))
    row.clear()
    return classified

def getConfusionMatrix(predictions,testSet,classes):
    matrix = pd.DataFrame(0,index=classes, columns=classes)
    for i in range(len(predictions)):
        correctClass= testSet[testSet.columns[-1]][i]
        if predictions[i] == correctClass: matrix.at[correctClass,correctClass]+=1
        else: matrix.at[correctClass,predictions[i]]+=1
    return matrix

def accurarcy(predictions, actuals):
    gotRight = 0
    for i in range(len(predictions)): 
        if (predictions[i] == actuals.values[i]): gotRight+=1
    return gotRight / len(predictions)

def precicion(Class,model):
    tp, all = model.at[Class, Class], model[Class].sum()
    return tp / all

def recall(Class,model):
    sum, tp = 0, model.at[Class,Class]
    for col in model: sum += model.at[Class, col]
    return tp / sum

def FScore(Class,model):
    return 2*(precicion(Class, model)*recall(Class, model))/(precicion(Class, model)+recall(Class, model))

trainCSVName, testCSVName, userSpecifiedK, confMatrix, pResults = cmdParameters()
trainCSV, testCSV = pd.read_csv(trainCSVName), pd.read_csv(testCSVName)
trainCSV.iloc[:,0:-1], testCSV.iloc[:,0:-1] = preprocessing.normalize(trainCSV.iloc[:,0:-1]), preprocessing.normalize(testCSV.iloc[:,0:-1])
classes = trainCSV[trainCSV.columns[-1]].unique()
predicted = iterate(trainCSV, testCSV , classes, userSpecifiedK)
if (pResults): 
    for i, data in enumerate(predicted): print("{0} {1}".format(i, data))
metrics = pd.DataFrame(0.0,index=["Precision","Recall","FScore"],columns=classes)
cMatrix = getConfusionMatrix(predicted,testCSV,classes)
for i, klass in enumerate(classes): metrics.at['Precision',klass], metrics.at['Recall', klass], metrics.at['FScore', klass] = precicion(klass,cMatrix), recall(klass, cMatrix),FScore(klass, cMatrix)
if(confMatrix): print("Accuracy of the model: {0}\nConfusion Matrix:\n{1} \n\nOther Metrics:\n{2}".format(accurarcy(predicted,testCSV.iloc[:,-1:]),cMatrix,metrics))
else: print("Accuracy of the model: {0}\nConfusion Matrix:\n{1}".format(accurarcy(predicted,testCSV.iloc[:,-1:]),cMatrix))