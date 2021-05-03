import numpy as np


def load_wine_dataset():
    path = 'Datasets/Wine.txt'
    dataset = np.genfromtxt(path,delimiter=',')
    return dataset


def load_glass_dataset():
    path = 'Datasets/Glass.txt'
    dataset = np.genfromtxt(path,delimiter='\t')
    dataset = dataset[:,:-1]
    return dataset


def load_sonar_dataset():
    path = 'Datasets/Sonar.txt'
    dataset = np.genfromtxt(path,delimiter=',',dtype=object)
    targets = list(set(dataset[:,-1]))
    for i in range(0,dataset.shape[0]):
        dataset[i,-1] = targets.index(dataset[i,-1])
    dataset = dataset.astype('float64')

    return dataset


def load_breastTissue_dataset():
    path = 'Datasets/BreastTissue.txt'
    dataset = np.genfromtxt(path,delimiter='\t')
    return dataset


def load_diabetes_dataset(): 
    dataset = np.genfromtxt('Datasets/Diabetes.txt',delimiter='\t')
    dataset = dataset[:,:-1]
    return dataset


def load_ionosphere_dataset():
    path = 'Datasets/Ionosphere.txt'
    dataset = np.genfromtxt(path,delimiter=',',dtype=object)
    targets = list(set(dataset[:,-1]))
    for i in range(0,dataset.shape[0]):
        dataset[i,-1] = targets.index(dataset[i,-1])
    dataset = dataset.astype('float64')

    return dataset
