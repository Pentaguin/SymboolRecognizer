'''
Requirements:
A neural network that can train to increase the accuracy to recognize X and O symbols.

Design Choice:

        ValueProvider
            /   \
        Link     Node
                 / \
          InputNode OutputNode     

Change WeightFactor from all links. Is it better? Remember it, but reset the weightFactor.
All the 18 links done? Update the weightFactor from the link which will give the best average error value
Start over again with link 0

Test during development:
- print values from all the outputnodes from 1 image
- print values from all the links from 1 image
- print values from the 2 outputnodes from 1 image
- print the error value from 1 image
- print the error values from all the images
- print the average value from all the images
- print the average values from the images with updated weightfactors

Final test:
Test 10 times. How many times is it correct?
10 times tested with 0.001 as target and everytime its accurate.
'''

import math
import data
import random as rnd

class NeuralNetwork():
    currentLinkIndex = 0
    previousWeightFactor = None
    currentImage = 0
    goalReached = False

    def createNetwork(self,inputNodeSize,outputNodeSize):
        self.inputNodeSize = inputNodeSize
        self.outputNodes = []
        inputNodes = []

        # Create 2 outputnodes.
        for outputNodeIndex in range(outputNodeSize):
            self.outputNodes.append(OutputNode())

        # Create 9 inputnodes. sqrt(9)
        for rowIndex in range(int(math.sqrt(inputNodeSize))):
            for colIndex in range(int(math.sqrt(inputNodeSize))):
                inputNodes.append(InputNode(rowIndex,colIndex,self))
        
        # Create 18 links. Each outputNode has 9 links.
        for inputNode in inputNodes:
            for outputNode in self.outputNodes:    
                Link(inputNode,outputNode)
  
    def train(self, targetValue, learningRate):
        self.learningRate = learningRate
        bestLinkIndex = 0
        bestAverageErrorValue = None

        while(not self.goalReached):  
            averageErrorValue =  self.getAverageErrorValue()
            
            if(bestAverageErrorValue is None or averageErrorValue < bestAverageErrorValue):  # New best average error value found. 
                bestAverageErrorValue = averageErrorValue
                bestLinkIndex = self.currentLinkIndex
            
            if(bestAverageErrorValue <= targetValue): # When average error value is smaller than the targetValue. 
                self.goalReached = True
            else: # AverageErrorValue has not reached the target.
                self.resetTemporaryWeightFactor()
                self.currentLinkIndex += 1
                
                # All the links are done. 
                if(self.currentLinkIndex > 17): 
                    self.currentLinkIndex = bestLinkIndex 
                    self.checkWhichWeightFactorIsBetter() # Update the weightFactor with the best link, which will give the best average error value.
                    self.currentLinkIndex = 0 # Set it back to 0 and start over again, but with 1 updated weight

                self.checkWhichWeightFactorIsBetter()
    
    def printResult(self):
        self.currentImage = 0

        # Itterate through all the images from the testSet and print the results
        for imageIndex in range(len(data.testSet)): 
            normalizedValue = self.normalize((self.outputNodes[0].getValue() , self.outputNodes[1].getValue()))     
            print(f"\nImage {self.currentImage}") 
            print(normalizedValue) 
            if(normalizedValue[0] > normalizedValue[1]):
                print("Symbol is O")
            else: 
                print("Symbol is X")
            self.currentImage += 1

    def checkWhichWeightFactorIsBetter(self):
        # Current weight + learningRate.
        self.changeWeightFactor(self.learningRate)
        averageValue1 = self.getAverageErrorValue()
        self.resetTemporaryWeightFactor()

        # Current weight - learningRate.
        self.changeWeightFactor(-self.learningRate)
        averageValue2 = self.getAverageErrorValue()
        self.resetTemporaryWeightFactor()

        # Compare which one is better and update the link with that weightFactor.
        if(averageValue1 < averageValue2):
            self.changeWeightFactor(self.learningRate)
        else:
            self.changeWeightFactor(-self.learningRate)

    # Set the new weight by getting the weight + weightFactorAddedValue.
    def changeWeightFactor(self,weightFactorAddedValue): 
        if(self.currentLinkIndex < self.inputNodeSize):
            self.previousWeightFactor = self.outputNodes[0].links[self.currentLinkIndex].weight
            self.outputNodes[0].links[self.currentLinkIndex].weight = self.outputNodes[0].links[self.currentLinkIndex].weight + weightFactorAddedValue
        else:
            # self.currentLinkIndex - 9, because 1 outputnode has only 9 links.
            self.previousWeightFactor = self.outputNodes[1].links[self.currentLinkIndex - self.inputNodeSize].weight
            self.outputNodes[1].links[self.currentLinkIndex - self.inputNodeSize].weight = self.outputNodes[1].links[self.currentLinkIndex - self.inputNodeSize].weight + weightFactorAddedValue

    # Reset the temporary weight.
    def resetTemporaryWeightFactor(self):
        if(self.previousWeightFactor != None):
            if(self.currentLinkIndex < self.inputNodeSize):
                self.outputNodes[0].links[self.currentLinkIndex].weight = self.previousWeightFactor
            else:
                self.outputNodes[1].links[self.currentLinkIndex - self.inputNodeSize].weight = self.previousWeightFactor

    def getAverageErrorValue(self):  
        sum = 0
        self.currentImage = 0
        imageSize = len(data.trainingSet)

        # Itterate through all the images from the trainingSet and calculate the error value.
        for imageIndex in range(imageSize): 
            normalizedValue = self.normalize((self.outputNodes[0].getValue() , self.outputNodes[1].getValue()))       
            target = data.outputDict.get('O') if data.trainingSet[self.currentImage][1] == "O" else data.outputDict.get('X')
            sum += self.calculateError(normalizedValue,target)
            self.currentImage += 1

        # Calculate average errorValue.
        return sum/imageSize

    def normalize(self, value):
        length = math.sqrt((value[0] * value[0]) + (value[1] * value[1]))
        return ((value[0] / length), (value[1] / length))

    def calculateError(self,value,target):
        return math.sqrt((value[0] - target[0]) ** 2)  + ((value[1] - target[1])  ** 2)
        
class ValueProvider():
    def getValue(self): # Pure virtual function
        raise Exception("abstract function call")

class Link(ValueProvider):
    def __init__(self,sourceNode,targetNode):
        self.weight = rnd.uniform(-1,1) # Random value from -1 to 1
        self.sourceNode = sourceNode
        targetNode.links.append(self)
        
    def getValue(self): 
        return self.weight * self.sourceNode.getValue() 
    
class Node(ValueProvider):
    pass

class InputNode(Node): 
    def __init__(self,x,y,neuralNetwork):
        self.x = x
        self.y = y
        self.neuralNetwork = neuralNetwork
    
    def getValue(self): # Return the pixel value     
        if(not self.neuralNetwork.goalReached):
            return data.trainingSet[self.neuralNetwork.currentImage][0][self.y][self.x]
        else:
            return data.testSet[self.neuralNetwork.currentImage][0][self.y][self.x]

class OutputNode(Node): 
    def __init__(self):
        self.links = []

    def getValue(self): 
        sum = 0
        
        for linkIndex in range(len(self.links)): 
            sum += self.links[linkIndex].getValue() # Sum all the link values.

        return math.exp(sum) # e^x as activation function

neuralNetwork = NeuralNetwork()
neuralNetwork.createNetwork(9,2)
neuralNetwork.train(0.001,0.1)
neuralNetwork.printResult()