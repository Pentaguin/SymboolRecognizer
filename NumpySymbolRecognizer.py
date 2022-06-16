import numpy as np
import data
import random as rnd
import math

class NeuralNetwork():
    currentLinkIndex = 0
    previousWeightFactor = None
    goalReached = False

    def __init__(self,inputSize,outputSize):
        self.layers = Layer(inputSize,outputSize)
        
    def train(self,trainingTarget,trainingData,learningRate):
        self.learningRate = learningRate
        bestLinkIndex = 0
        bestAverageErrorValue = None

        while(not self.goalReached):
            averageErrorValue =  self.getAverage(trainingData)
            
            if(bestAverageErrorValue is None or averageErrorValue < bestAverageErrorValue):  # New best average error value found. 
                bestAverageErrorValue = averageErrorValue
                bestLinkIndex = self.currentLinkIndex
            
            if(bestAverageErrorValue <= trainingTarget): # When average error value is smaller than the trainingTarget
                self.goalReached = True
                break
            else: # AverageErrorValue has not reached the target.
                self.resetTemporaryWeightFactor()
                self.currentLinkIndex += 1
                
                # All the links are done. 
                if(self.currentLinkIndex > 17): 
                    self.currentLinkIndex = bestLinkIndex 
                    self.checkWhichWeightFactorIsBetter(trainingData) # Update the weightFactor with the best link, which will give the best average error value.
                    self.currentLinkIndex = 0 # Set it back to 0 and start over again, but with 1 updated weight

                self.checkWhichWeightFactorIsBetter(trainingData)

    def printTestResult(self,testData):
        imageIndex = 0

        for image,target in testData:
            normalizedValue = self.normalize((self.getValue(image)[0] , self.getValue(image)[1])) 
            print(f"\nIMAGE {imageIndex}")
            print(normalizedValue)
            if(normalizedValue[0] > normalizedValue[1]):
                print("Symbol is O")
            else: 
                print("Symbol is X")           
            imageIndex += 1
                  
    def getAverage(self,trainingData):
        sum = 0

        for image,target in trainingData:
            normalizedValue = self.normalize((self.getValue(image)[0] , self.getValue(image)[1]))       
            sum += self.calculateError(normalizedValue,target)

        return sum/len(trainingData)

    def getValue(self,inputVector):
        return self.layers.getValue(inputVector)

    def normalize(self, value):
        length = math.sqrt((value[0] * value[0]) + (value[1] * value[1]))
        return ((value[0] / length), (value[1] / length))

    def calculateError(self,value,target):
        return (math.sqrt((value[0] - target[0]) ** 2)  + ((value[1] - target[1])  ** 2))

    def checkWhichWeightFactorIsBetter(self,trainingData):
        # Current weight + learningRate.
        self.changeWeightFactor(self.learningRate)
        averageValue1 = self.getAverage(trainingData)
        self.resetTemporaryWeightFactor()

        # Current weight - learningRate.
        self.changeWeightFactor(-self.learningRate)
        averageValue2 = self.getAverage(trainingData)
        self.resetTemporaryWeightFactor()

        # Compare which one is better and update the link with that weightFactor.
        if(averageValue1 < averageValue2):
            self.changeWeightFactor(self.learningRate)
        else:
            self.changeWeightFactor(-self.learningRate)

    # Set the new weight by getting the weight + weightFactorAddedValue.
    def changeWeightFactor(self,weightFactorAddedValue):
        if(self.currentLinkIndex < 9):
            self.previousWeightFactor = self.layers.weight[0][self.currentLinkIndex]
            self.layers.weight[0][self.currentLinkIndex] =  self.layers.weight[0][self.currentLinkIndex] + weightFactorAddedValue
        else:
            self.previousWeightFactor = self.layers.weight[1][self.currentLinkIndex - 9]
            self.layers.weight[1][self.currentLinkIndex - 9] =  self.layers.weight[1][self.currentLinkIndex - 9] + weightFactorAddedValue

    # Reset the temporary weight.
    def resetTemporaryWeightFactor(self):
        if(self.previousWeightFactor != None):
            if(self.currentLinkIndex < 9):
                self.layers.weight[0][self.currentLinkIndex] =  self.previousWeightFactor
            else:
                self.layers.weight[1][self.currentLinkIndex - 9] =  self.previousWeightFactor

class Layer():
    def __init__(self,inputSize,outputSize):
        self.weight = np.random.rand(outputSize,inputSize)

    def getValue(self,inputVector):
        return self.sigmoid(self.applyWeightFactors(inputVector))

    def applyWeightFactors(self,inputVector):
        return np.dot(self.weight,inputVector)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

def parse_dataset(dataset): # Split the 0,1 and the X,O
    return [
        (np.array(image).flatten(), np.array(data.outputDict[target]))
        for image, target in dataset
    ]

neuralNetwork = NeuralNetwork(9,2)
neuralNetwork.train(0.1,parse_dataset(data.trainingSet),0.1)
neuralNetwork.printTestResult(parse_dataset(data.testSet))





