import numpy as np
from sklearn.tree import DecisionTreeClassifier
from EnsembleBase import Ensemble

class AdaBoostModel(Ensemble):

    def __init(self,classifier_counts=10):
        self.classifier_counts = classifier_counts

    def train(self,traindataset):
        pass

    def classification(self,x):
        pass


if __name__ == '__main__':
    test = AdaBoostModel()
    data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    print(data)
    sdata = test.bootstrap(data)
    print(sdata)
