from sklearn.tree import DecisionTreeClassifier
import numpy as np
from EnsembleBase import Ensemble


# Bootstrap Aggregation Model
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
        #print(f'Correct Predict {correct_predict}, Wrong Predict {wrong_predict}')
        calculated_accuray = (correct_predict / len(testdata)) * 100
        #print(f'Accuracy is {calculated_accuray}')
        return calculated_accuray

def add_noise(dataset,percent):
    column_count = dataset.shape[1] - 1
    column_noise_count = round(column_count * percent)

    selected_columns = []
    count = 0

    while count < column_noise_count :
        random_column = np.random.randint(0,column_count)
        if random_column in selected_columns :
            continue
        selected_columns.append(random_column)
        gaussain_noise = np.random.normal(size=len(dataset[:,random_column]))
        dataset[:,random_column] += gaussain_noise
        count += 1
    return dataset


def bagging_test_model(dataset,k_run=1,noise_percent=0):
    train_dataset_percentage = 0.7
    train_dataset_size = round(len(dataset) * train_dataset_percentage)

    accuracies = []

    for i in range(0,k_run):
        shuffled_dataset = np.copy(dataset)
        np.random.shuffle(shuffled_dataset)
        train_dataset = shuffled_dataset[0:train_dataset_size]
        test_dataset = shuffled_dataset[train_dataset_size:len(shuffled_dataset)]
        model = BaggingModel()

        if noise_percent > 0 :
            train_dataset = add_noise(train_dataset,noise_percent)

        model.train(train_dataset)
        calculated_accuracy = model.accuracy(test_dataset)
        accuracies.append(calculated_accuracy)

    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print(f'Mean/Standard Deviation of {k_run} Run is {mean}/{std}')


if __name__ == '__main__':
    print('Testing Bagging Model With Wine Dataset')
    dataset = np.genfromtxt('Datasets/Wine.txt',delimiter=',')
    bagging_test_model(dataset,10)
