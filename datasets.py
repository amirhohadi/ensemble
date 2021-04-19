import numpy as np


def load_wine_dataset():
    path = 'Datasets/Wine.txt'
    dataset = np.genfromtxt(path,delimiter=',')
    print(dataset.shape)
    x = dataset[:,:-1]
    y = dataset[:,-1]
    print(f'{x.shape}  {y.shape}')
    return x,y


def load_glass_dataset():
    path = 'Datasets/Glass.txt'
    dataset = np.genfromtxt(path,delimiter='\t')
    print(dataset.shape)
    x = dataset[:,:-1]
    y = dataset[:,-1]
    print(f'{x.shape}  {y.shape}')
    return x,y

def load_sonar_dataset():
    path = 'Datasets/Sonar.txt'
    dataset = np.genfromtxt(path,delimiter=',',dtype=object)
    print(dataset.shape)
    print(dataset)
    x = dataset[:,:-1]
    y = dataset[:,-1]
    print(f'{x.shape}  {y.shape}')
    return x,y
