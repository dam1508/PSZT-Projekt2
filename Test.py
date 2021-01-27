
from time import time
from AdaBoost import AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import numpy as np

option = input("Nadpisać logi? t/n: ")
if option == "t" or option == "T" :
    ourLog = open("logOurs.txt", "w")
    skLog = open("logSkLearn.txt", "w")
else:
    ourLog = open("logOurs.txt", "a")
    skLog = open("logSkLearn.txt", "a")

option = input("Wiele opcji drzew? t/n: ")
if option == 't' or option == 'T' :
    min = int(input("Minimalna ilość drzew: "))
    max = int(input("Maksymalna ilość drzew: "))+1
else :
    min = int(input("Podaj liczbę drzew: "))
    max = min+1

# Wczytanie danych
test = np.loadtxt('data.txt', delimiter=",")
dat = test[:, 0:13]
train = np.loadtxt('learning.txt', delimiter=",")
lrn = train[:, 8]

# Dane dla sklearn
skTrainParam, skTestParam, skTrainOut, skTestOut = train_test_split(dat, lrn, test_size=0.2)

# Dane dla naszego algorytmu
trainParam, testParam, trainOut, testOut = train_test_split(dat, lrn, test_size=0.2)

trainParam=trainParam.transpose()
trainOut[trainOut == 1] = 1
trainOut[trainOut == 0] = -1

testParam=testParam.transpose()
testOut[testOut == 1] = 1
testOut[testOut == 0] = -1


for estimators in range(min, max):
    our = AdaBoost(trainParam, trainOut)
    sk = AdaBoostClassifier(n_estimators = estimators, learning_rate=1)

    start = time()
    model = sk.fit(skTrainParam, skTrainOut)
    end = time()
    skTime = end - start

    start = time()
    our.simulate(estimators)
    end = time()
    ourTime = end - start

    #Prognoza odpoweidzi dla zestawu danych
    skPred = model.predict(skTestParam)
    ourPred = our.estimate(testParam)

    ourLog.write(f"Stump: {estimators} Time: {ourTime} Precision: {metrics.accuracy_score(testOut, ourPred)}\n")
    skLog.write(f"Stump: {estimators} Time {skTime} Precision: {metrics.accuracy_score(skTestOut, skPred)}\n")
    print (f"Ilosc stumpow: {estimators}")
    print (f"Czas naszego algorytmu: {ourTime} s")
    print ("Dokladnosc naszego algorytmu:", metrics.accuracy_score(testOut, ourPred))

    print (f"Czas sklearn: {skTime} s")
    print ("Dokladnosc sklearn:", metrics.accuracy_score(skTestOut, skPred))
    print ("---------------------------------")

ourLog.close()
skLog.close()
