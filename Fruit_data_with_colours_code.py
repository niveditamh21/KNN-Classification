#Python3 program to find groups of unknown points using K nearest neighbour algorithm.
#Using library methods and splittind datasets into trained and test set

import pandas as pd
from sklearn.model_selection import train_test_split

#Euclidean Distance
import math
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
  
#Obtaining k nearest neighbours
import operator 
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#Classifying by counting frequencies
import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#Finding accuracy of predictions made
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(predictions)):
        if testSet[x][-1] == predictions[x]: 
            correct = correct + 1
            
    return (correct/float(len(predictions))*100) 

#Main function
dataset=pd.read_csv("fruit_data_with_colours.csv")
features=["mass","width","height","color_score","fruit_name"]
X=dataset[features].values
trainSet,testSet = train_test_split(X, test_size=0.25, random_state=0)
k=5
n=getNeighbors(X,[80,5.8,4.3,0.77],k)
print('The value is classified to ' + getResponse(n))
predictions=[]
for i in range(len(testSet)):
    n=getNeighbors(X,testSet[i],k)
    #print(getResponse(n))
    result=getResponse(n)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet[i][-1]))
print('Accuracy is '+ repr(getAccuracy(testSet,predictions)) + '%')
#print(getResponse(n))
