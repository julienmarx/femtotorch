class VanillaSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.lr = learning_rate
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()