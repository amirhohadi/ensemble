from sklearn.tree import DecisionTreeClassifier
import numpy as np
from EnsembleBase import Ensemble

#Bootstrap Aggregation Model
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
        outputs = []
        for i in range(0,self.classifier_counts):
            out = self.hypothesises[i].predict(x.reshape((1,-1)))
            outputs.append(out[0])
        #print(outputs)
        return max(set(outputs),key=outputs.count)

    def bootstrap(self,data):
        bootstraped_data = np.zeros(data.shape)

        for i in range(0,len(data)):
            rand_index = np.random.randint(0,len(data))
            bootstraped_data[i,:] = data[rand_index,:]

        return bootstraped_data

    def accuracy(self,testdata):
        correct_predict = 0
        wrong_predict = 0
        for i in range(0,len(testdata)):
            yhat = self.classification(testdata[i,:-1])
            y = testdata[i,-1]
            if yhat == y :
                correct_predict += 1
            else:
                wrong_predict += 1
        print(f'Correct Predict {correct_predict}, Wrong Predict {wrong_predict}')
        calculated_accuray = (correct_predict / len(testdata)) * 100
        print(f'Accuracy is {calculated_accuray}')


if __name__ == '__main__':
    print('Testing Bagging Model With Glass Dataset')
    dataset = np.genfromtxt('Datasets/Wine.txt',delimiter=',')
    model = BaggingModel()
    model.train(dataset)
    testdataset = dataset[:,:-1]
    model.accuracy(dataset)

