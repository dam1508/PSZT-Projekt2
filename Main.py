from time import time

import numpy as np
from AdaBoost import AdaBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Wczytanie danych
data = np.loadtxt('data.txt', delimiter=",")
dat = data[:, 0:13]

learning = np.loadtxt('learning.txt', delimiter=",")
lrn = learning[:, 8]

# Podział i przygotowanie danych
trainingParameters, testingParameters, trainingOutcome, testingOutcome = train_test_split(dat, lrn, test_size=0.3)

trainingParameters = trainingParameters.transpose()
trainingOutcome[trainingOutcome == 1] = 1
trainingOutcome[trainingOutcome == 0] = -1

testingParameters = testingParameters.transpose()
testingOutcome[testingOutcome == 1] = 1
testingOutcome[testingOutcome == 0] = -1

# Zastosowanie algorytmu
adaBoost = AdaBoost(trainingParameters, trainingOutcome)
start = time()
adaBoost.simulate(10)
end = time()
timeElapsed = end - start

# Estymacja
outcomeEstimation = adaBoost.estimate(testingParameters)

print(f"Czas dzialania algorytmu: {timeElapsed} s")
print ("Estymacja:", len(outcomeEstimation[outcomeEstimation == testingOutcome]))
print ("Precyzja:", accuracy_score(testingOutcome, outcomeEstimation))

