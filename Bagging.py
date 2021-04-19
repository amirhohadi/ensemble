from sklearn.tree import DecisionTreeClassifier
import numpy as np
from EnsembleBase import Ensemble

class BaggingModel(Ensemble):

    def __init__(self,classifier_counts=10):
        self.classifier_counts = classifier_counts

    def train(self,traindataset):
        self.hypothesises = []
        for i in range(0,self.classifier_counts):
            hypothesis = DecisionTreeClassifier()
            bootstraped_data = self.bootstrap(traindataset)
            x = bootstraped_data[:,:-1]
            y = bootstraped_data[:,-1]
            hypothesis.fit(x,y)
            self.hypothesises.append(hypothesis)

    def classification(self,x):
        pass

    def bootstrap(self,data):
        bootstraped_data = np.zeros(data.shape)

        for i in range(0,len(data)):
            rand_index = np.random.randint(0,len(data))
            bootstraped_data[i,:] = data[rand_index,:]

        return bootstraped_data

