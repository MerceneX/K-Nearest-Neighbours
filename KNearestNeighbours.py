import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split




### WINE
"""trainData = pd.read_csv('wine_ucna.csv')
testData = pd.read_csv('wine_testna.csv')

XTrain = np.array(trainData.drop(['Quality'],1))
yTrain = np.array(trainData['Quality'])

XTest = np.array(testData.drop(['Quality'],1))
yTest = np.array(testData['Quality'])"""

### IRIS
trainData = pd.read_csv('iris_ucna.csv')
testData = pd.read_csv('iris_testna.csv')

XTrain = np.array(trainData.drop(['Species'],1))
yTrain = np.array(trainData['Species'])

XTest = np.array(testData.drop(['Species'],1))
yTest = np.array(testData['Species'])

clf = neighbors.KNeighborsClassifier()
clf.fit(XTrain,yTrain)

accuracy = clf.score(XTest,yTest)
print(accuracy)

example = np.array([5,2.7,4.3,1.2])
example = example.reshape(len(example),-1)

prediction = clf.predict(example)
print(prediction)