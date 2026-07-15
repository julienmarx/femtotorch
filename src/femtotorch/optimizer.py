import numpy as np 
from abc import ABC, abstractmethod


class Optimizer(ABC):
    
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.lr = learning_rate

    @abstractmethod
    def step(self):
        """Updates the weights of the net"""

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is None: 
                continue
            p.zero_grad()
        
    def get_lr(self):
        return self.lr
    
    def set_lr(self, lr):
        self.lr = lr


class VanillaSGD(Optimizer):

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
    


class SGD_Moment(Optimizer):
    
    def __init__(self, parameters, learning_rate, weight_decay = 5e-4, momentum=0.9): #weight_decay default for cifar10
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # eager initialization to avoid checking "if initialize" in the traning loop
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]


    def step(self):
        for p, v in zip(self.parameters, self.velocities): #zip() to pair (self.parameters[i], self.velocities[i])
            # to avoid memorization (big weights), we use weight decay
            # we want to minimize WL = L(p) + (λ/2)·‖p‖², dWL/dp = p.grad + λ * p.data
            g = p.grad + self.weight_decay * p.data
            
            v[:] = self.momentum * v + g # [:] so the velocity list keeps seing the updated array in the same reference
            p.data -= self.lr * v



class CosineScheduler:

    def __init__(self, optimizer: Optimizer, total_epochs, lr_min=0.0):
        self.optimizer = optimizer # store the optimizer object reference
        self.lr_min = lr_min
        self.lr_max = optimizer.get_lr()
        self.total_epochs = total_epochs
        self.epoch = 0

    def step(self):
        self.epoch += 1
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * (self.epoch/self.total_epochs)))
        self.optimizer.set_lr(lr)



    