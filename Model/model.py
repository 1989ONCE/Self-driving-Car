import abc
import numpy as np

class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.train_result = []
        self.test_result = []
        self.weight = np.array([])

        # member that stores the best experiment result
        self.best_weight = []
        self.best_epoch = 0
        self.best_train_acc = 0.0
        self.best_test_acc = 0.0
        self.class_list = []

    @abc.abstractmethod
    def train(self):
        return NotImplemented