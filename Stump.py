import numpy as np
import math


class Stump:
    def __init__(self, parameters, outcome):
        self.parameters = np.array(parameters)
        self.outcome = np.array(outcome)
        self.entries = self.parameters.shape[0]
    
    def calculateTotalError(self, weight, steps = 100): # Znalezienie najmniejszej ilosci bledow
        
        min = math.inf 
        thresholdValue = 0;
        thresholdPosition = 0;
        thresholdFlag = 0;
        self.weight = np.array(weight)

        for i in range(self.entries):
            value, errorCount = self.findThreshold(i, 1, steps)

            if (errorCount < min):
                min = errorCount
                thresholdValue = value
                thresholdPosition = i
                thresholdFlag = 1

        for i in range(self.entries):  
            wartosc, liczbaBledow= self.findThreshold(i, -1, steps)

            if (errorCount < min):
                min = errorCount
                thresholdValue = value
                thresholdPosition = i
                thresholdFlag = -1

        self.thresholdValue = thresholdValue
        self.thresholdPosition = thresholdPosition
        self.thresholdFlag = thresholdFlag

        return min
    
    def findThreshold(self, idx, flag, steps):  # Znalezienie progu i odpowiedniej flagi

        threshold = 0
        value = 0
        lowerBound = np.min(self.parameters[idx, :])  # Znalezienie najmniejszego atrybutu posrod danych parametrow
        upperBound = np.max(self.parameters[idx, :])  # Znalezienie najwiekszego atrybutu posrod danych parametrow
        minErrors = math.inf                          
        interval = (upperBound - lowerBound) / steps  # Obliczenie skoku

        for threshold in np.arange(lowerBound, upperBound, interval):
            currEstimation = self.estimateData(self.parameters, idx, threshold, flag).transpose()
            wrongCount = np.sum((currEstimation != self.outcome) * self.weight)                     # Znalezienie ilosci bledow w estymacji

            if wrongCount < minErrors:
                minErrors = wrongCount
                value = threshold                     # Jesli liczba bledow jest mniejsza, zapamietujemy obecna wartosc

        return value, minErrors

    
    def estimateData(self, givenParameters, idx, threshold, tag): # Estymacja Stumpu o przekazanych danych

        givenParameters = np.array(givenParameters).reshape(self.entries, -1)
        estimatedOutcome = np.ones((np.array(givenParameters).shape[1], 1))
        estimatedOutcome[givenParameters[idx, :] * tag < threshold * tag] = -1

        return estimatedOutcome


    def estimateStump(self, givenParameters): # Estymacja obecnego Stumpu

        givenParameters = np.array(givenParameters).reshape(self.entries, -1)
        estimatedOutcome = np.ones((np.array(givenParameters).shape[1], 1))
        estimatedOutcome[givenParameters[self.thresholdPosition, :] * self.thresholdFlag < self.thresholdValue * self.thresholdFlag] = -1

        return estimatedOutcome
	
