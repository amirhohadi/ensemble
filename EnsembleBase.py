import abc

class Ensemble(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self,traindataset):
        pass

    @abc.abstractmethod
    def classification(self,x):
        pass
