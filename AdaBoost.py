import numpy as np
from sklearn.tree import DecisionTreeClassifier
from EnsembleBase import Ensemble

class AdaBoostModel(Ensemble):

    def __init(self,max_depth,classifier_counts=10):
        self.classifier_counts = classifier_counts
        self.max_depth = max_depth

    def train(self,traindataset):
        self.hypothesises = []
        self.alpha_t = []

        weights = np.repeat(1 / len(traindataset) , len(traindataset))  # Initializing W1 with 1/m

        for i in range(0,self.classifier_counts):
            model = DecisionTreeClassifier(max_depth=self.max_depth)    # Classifier Model D(k)
            x = traindataset[:,:-1]
            y = traindataset[:,-1]
            model.fit(x,y,sample_weight=weights)    # Train Weak Classifier

            baseclassifier_classification_outputs = model.predict(x)  # Get Outputs of Model for Training DataSet
            # Calculate Error E(k)
            e_k = 0
            Ls = []
            for index,output in enumerate(baseclassifier_classification_outputs):
                L_output = 1 if output != y[index] else 0
                Ls.append(L_output)
                e_k += weights[index] * L_output

            if e_k == 0 or e_k >= 0.5 :
                weights = np.repeat(1 / len(traindataset) , len(traindataset))
                continue

            alpha_k = e_k / (1 - e_k)
            self.alpha_t.append(alpha_k)
            self.hypothesises.append(model)

            #Update Weights W(k+1)
            sigma = 0
            for i in range(0,len(traindataset)):
                sigma += weights[i] * (alpha_k ** (1 - Ls[i]))

            for j in range(0,len(traindataset)):
                weights[j] = (weights[j] * (alpha_k ** (1 - Ls[j]))) / sigma

    def classification(self,x):
        pass


if __name__ == '__main__':
    pass
