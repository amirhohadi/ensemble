import numpy as np
from sklearn.tree import DecisionTreeClassifier
from EnsembleBase import Ensemble
import math

class AdaBoostModel(Ensemble):

    def __init__(self,max_depth,classifier_counts=10):
        self.classifier_counts = classifier_counts
        self.max_depth = max_depth

    def train(self,traindataset):
        self.hypothesises = []
        self.beta_t = []

        weights = np.repeat(1 / len(traindataset) , len(traindataset))  # Initializing W1 with 1/m

        for classifier_index in range(0,self.classifier_counts):
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            x = traindataset[:,:-1]
            y = traindataset[:,-1]
            model.fit(x,y,sample_weight=weights)    # Train Weak Classifier

            base_classifier_output = model.predict(x)

            calculated_error = 0
            Lk = []
            for index,output in enumerate(base_classifier_output):
                l_output = 1 if output != y[index] else 0
                Lk.append(l_output)
                calculated_error += weights[index] * l_output

            #print(f'Error of {classifier_index} is {calculated_error}')

            if calculated_error == 0 or calculated_error >= 0.5:
                continue

            beta_k = calculated_error / (1 - calculated_error)

            #print(f'Beta(k) is {beta_k}')

            self.hypothesises.append(model)
            self.beta_t.append(beta_k)

            #Updating Weights(W(k+1,j))

            sigma = 0   # Makhraj Kasr
            for i in range(0,len(traindataset)):
                sigma += weights[i] * (beta_k ** (1 - Lk[i]))

            for i in range(0,len(traindataset)):
                weights[i] = (weights[i] * (beta_k ** (1 - Lk[i]))) / sigma

    def classification(self,x):
        outputs = []
        x = x.reshape((1,-1))
        for hypothesis in self.hypothesises:
            out = hypothesis.predict(x)
            outputs.append(out[0])

        mu_s = {}
        bt = np.array(self.beta_t)

        for label in set(outputs):
            model_predicts_this_label = list(map(lambda x : 1 if x == label else 0,outputs))

            temp = model_predicts_this_label * bt
            temp = [x for x in temp if x != 0]
            temp = list(map(lambda x : math.log(1 / x),temp))
            mu = sum(temp)
            mu_s[label] = mu

        best_mu = max(mu_s,key=mu_s.get)
        #print(f'{best_mu} is selected with Mu {mu_s[best_mu]}')
        return best_mu

    def accuracy(self,testdata):
        correct_predict = 0
        wrong_predict = 0
        for index in range(0,len(testdata)):
            yhat = self.classification(testdata[index,:-1])
            y = testdata[index,-1]
            if yhat == y :
                correct_predict += 1
            else :
                wrong_predict += 1
        print(f'Correct Predict {correct_predict} and Wrong Predict {wrong_predict}')
        calculated_accuracy = (correct_predict / len(testdata)) * 100
        print(f'Accuracy is {calculated_accuracy}')
        return calculated_accuracy

def adaboost_test_model(dataset,max_depth,k_run=1):
    train_dataset_percentage = 0.7
    train_dataset_size = round(len(dataset) * train_dataset_percentage)

    accuracies = []

    for i in range(0,k_run):
        shuffled_dataset = np.copy(dataset)
        np.random.shuffle(shuffled_dataset)

        train_dataset = shuffled_dataset[0:train_dataset_size]
        test_dataset = shuffled_dataset[train_dataset_size:len(dataset)]

        model = AdaBoostModel(max_depth=max_depth)

        model.train(train_dataset)
        calculated_accuracy = model.accuracy(test_dataset)
        accuracies.append(calculated_accuracy)

    std = np.std(accuracies)
    print(f'Standard Deviation of {k_run} is {std}')


if __name__ == '__main__':
    print('Testing Model With Diabetes Dataset')
    dataset = np.genfromtxt('Datasets/Diabetes.txt',delimiter='\t')
    dataset = dataset[:,:-1]
    adaboost_test_model(dataset,2,10)
