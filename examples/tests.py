

import femtotorch as ft
import numpy as np
import time

class OptiCnnNet:
    
    def __init__(self):
        self.batch_size = 64

        self.conv1 = ft.OptiConv2d(in_channels=3, out_channels=32, kernel_size=3, stride =1, padding=1)
        self.conv2 = ft.OptiConv2d(in_channels = 32, out_channels=32, kernel_size=3, stride = 2, padding=1)
        self.conv3 = ft.OptiConv2d(in_channels=32, out_channels=64, kernel_size=3, stride =1, padding=1) 
        self.conv4 = ft.OptiConv2d(in_channels = 64, out_channels=64, kernel_size=3, stride = 2, padding=1)
        self.out_conv4 = self.conv4.size_map(16, 16) # since conv3 output is 64 * 16 * 16
        
        self.model = ft.MLP(self.out_conv4, [256, 10]) 

    def __call__(self, X):

        x = self.conv1(X.reshape(-1, 3, 32, 32)).relu() # the -1 allows flexibility on the last batch 

        x = self.conv2(x).relu()

        x = self.conv3(x).relu()

        x = self.conv4(x).relu()
        x = x.reshape(-1, self.out_conv4)

        soft_out = ft.softmax(self.model(x))

        return soft_out
        
    
    def parameters(self):
        return [*self.model.parameters(), *self.conv1.parameters(), *self.conv2.parameters(), *self.conv3.parameters(), *self.conv4.parameters()]
 

class MaxPoolCnnNet:
    
    def __init__(self):
        self.batch_size = 128

        self.conv1 = ft.OptiConv2d(in_channels=3, out_channels=32, kernel_size=3, stride =1, padding=1) 
        self.out_conv1 = self.conv1.size_map(32, 32)
        self.pool1 = ft.MaxPool2d(kernel_size=2)

        self.conv2 = ft.OptiConv2d(in_channels = 32, out_channels=64, kernel_size=3, stride = 1, padding=1)
        self.out_conv2 = self.conv2.size_map(16, 16) # since pool1 output is 32 * 16 * 16
        self.pool2 = ft.MaxPool2d(kernel_size=2)

        self.conv3 = ft.OptiConv2d(in_channels=64, out_channels=128, kernel_size=3, stride =1, padding=1) 
        self.out_conv3 = self.conv3.size_map(8, 8) # since pool2 output is 32 * 8 * 8
        self.pool3 = ft.MaxPool2d(kernel_size=2)
        self.out_pool3 = (self.out_conv3 // 4)
        self.model = ft.MLP(self.out_pool3, [128, 10]) 

    def __call__(self, X):

        x = self.conv1(X.reshape(-1, 3, 32, 32)).relu() # the -1 allows flexibility on the last batch 
        x = self.pool1(x)

        x = self.conv2(x).relu()
        x = self.pool2(x)

        x = self.conv3(x).relu()
        x = self.pool3(x)

        x = x.reshape(-1, self.out_pool3)

        soft_out = ft.softmax(self.model(x))

        return soft_out
        
    
    def parameters(self):
        return [*self.model.parameters(), *self.conv1.parameters(), *self.conv2.parameters(), *self.conv3.parameters()]


Xtrain, Ytrain, Xtest, Ytest = ft.load_cifar10("data/cifar10")
net2 = MaxPoolCnnNet()
params_list2 = net2.parameters()


start = time.perf_counter()

# load weights used in 1st inference by net into net2
ft.load("checkpoints_weights/CIFAR_CNN.npz", params_list2)

end = time.perf_counter()
print(f"load: {end - start:.4f} seconds")

start = time.perf_counter()

pred = net2(ft.Tensor(Xtest[:1000])).argmax(axis=-1)
accuracy = (pred.data[:1000] == Ytest[:1000]).mean()

end = time.perf_counter()
print(f"inference: {end - start:.4f} seconds")


start = time.perf_counter()
with ft.no_grad():
    pred = net2(ft.Tensor(Xtest)).argmax(axis=-1)
    accuracy = (pred.data == Ytest).mean()

end = time.perf_counter()
print(f"inference no grad: {end - start:.4f} seconds")

print(f"test accuracy: {accuracy}")
