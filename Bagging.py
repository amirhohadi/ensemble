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
        return calculated_accuray

def bagging_test_model(dataset,k_run=1):
    train_dataset_percentage = 0.7
    train_dataset_size = round(len(dataset) * train_dataset_percentage)

    accuracies = []

    for i in range(0,k_run):
        shuffled_dataset = np.copy(dataset)
        np.random.shuffle(shuffled_dataset)
        train_dataset = shuffled_dataset[0:train_dataset_size]
        test_dataset = shuffled_dataset[train_dataset_size:len(shuffled_dataset)]
        model = BaggingModel()
        model.train(train_dataset)
        calculated_accuracy = model.accuracy(test_dataset)
        accuracies.append(calculated_accuracy)

    std = np.std(accuracies)
    print(f'Standard Deviation of {k_run} Run is {std}')


if __name__ == '__main__':
    print('Testing Bagging Model With Wine Dataset')
    dataset = np.genfromtxt('Datasets/Wine.txt',delimiter=',')
    bagging_test_model(dataset,10)
