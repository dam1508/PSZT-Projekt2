import numpy as np
from Stump import Stump
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, parameters, outcome, stump = Stump):
        self.parameters = np.array(parameters)
        self.outcome = np.array(outcome).flatten('F')
        self.stump = stump
        self.sums = np.zeros(self.outcome.shape)
        self.weight = np.ones((self.parameters.shape[1],1)).flatten('F')/self.parameters.shape[1]   # Na poczatku waga jest rozlozona rownomiernie
        self.stumpCount = 0                 # Obecna liczba klasyfikatowrow  w AdaBoost
        self.learners = {}                  # Lista tych klasyfikatorow
        self.aos = {}                       # Amount of Say
        
    def simulate(self, maxLearners = 10):

        for i in range(maxLearners):
            self.learners.setdefault(i)
            self.aos.setdefault(i)

        for i in range(maxLearners):
            self.learners[i] = self.stump(self.parameters, self.outcome)
            totalError = self.learners[i].calculateTotalError(self.weight)
            self.aos[i] = 1.0 / 2*np.log((1 - totalError) / totalError)                         # Liczenie Amount of Say ze wzoru
            result = self.learners[i].estimateStump(self.parameters)

            newWeight = self.weight * np.exp(-self.aos[i] * self.outcome * result.transpose())  # Obliczenie nowej wagi
            self.weight = (newWeight / newWeight.sum()).flatten('F')                            # Oraz jej normalizacja (Aby wszystkie sumowaly sie do 1)
            self.stumpCount = i

            if (self.wrongGuesses(i) == 0):     # W przypadku zanlezienia idealnej kombinacji, konczymy procedure
                break

    def wrongGuesses(self, idx):    # Sprawdzenie ile bledow zostalo popelnionych

        self.sums = self.sums + self.learners[idx].estimateStump(self.parameters).flatten('F') * self.aos[idx]
        
        estimatedOutcome = np.zeros(np.array(self.sums).shape)
        estimatedOutcome[self.sums >= 0] = 1
        estimatedOutcome[self.sums < 0] = -1
        
        wrongCount = (estimatedOutcome != self.outcome).sum()
        return wrongCount
    
    def estimate(self, testParameters):  # Estymacja Adaboostu

        testParameters = np.array(testParameters)
        totalSum = np.zeros(testParameters.shape[1])

        for i in range(self.stumpCount + 1):
            totalSum = totalSum + self.learners[i].estimateStump(testParameters).flatten('F') * self.aos[i]

        estimatedOutcome = np.zeros(np.array(totalSum).shape)
        estimatedOutcome[totalSum >= 0] = 1
        estimatedOutcome[totalSum < 0] = -1
        return estimatedOutcome
