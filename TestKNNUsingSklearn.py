import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import sys



### WINE
trainData = pd.read_csv('wine_ucna.csv')
testData = pd.read_csv('wine_testna.csv')

XTrain = np.array(trainData.drop(['Quality'],1))
yTrain = np.array(trainData['Quality'])

XTest = np.array(testData.drop(['Quality'],1))
yTest = np.array(testData['Quality'])

### IRIS
"""trainData = pd.read_csv('iris_ucna.csv')
testData = pd.read_csv('iris_testna.csv')

XTrain = np.array(trainData.drop(['Species'],1))
yTrain = np.array(trainData['Species'])

XTest = np.array(testData.drop(['Species'],1))
yTest = np.array(testData['Species'])"""

### Make model
clf = neighbors.KNeighborsClassifier()
clf.fit(XTrain,yTrain)

accuracy = clf.score(XTest,yTest)
print(accuracy)

#Example Wine
example = np.array([[14.12,1.48,2.32,16.8,95,2.2,2.43,0.26,1.54,5,1.15,2.81,1279]])
example = example.reshape(len(example),-1)

#Example Iris
"""example = np.array([[5,2.7,4.3,1.2]])
example = example.reshape(len(example),-1)"""

prediction = clf.predict(example)
print(prediction)