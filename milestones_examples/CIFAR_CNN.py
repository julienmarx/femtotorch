
import femtotorch as ft
import numpy as np


class OptiCnnNet:
    
    def __init__(self):
        self.batch_size = 64
        self.conv1 = ft.Opti_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride =1, padding=1) 
        self.out_conv1 = self.conv1.size_map(32, 32)

        self.conv2 = ft.Opti_Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride = 2, padding=1)
        self.out_conv2 = self.conv2.size_map(32, 32) # since conv1 output is 32 * 32 * 32

        self.conv3 = ft.Opti_Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride =1, padding=1) 
        self.out_conv3 = self.conv3.size_map(16, 16) # since conv2 output is 32 * 16 * 16

        self.conv4 = ft.Opti_Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride = 2, padding=1)
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
 

# Initialization
Xtrain, Ytrain, Xtest, Ytest = ft.load_cifar10("data/cifar10")
net = OptiCnnNet()
params_list = net.parameters()
gradient_updater = ft.VanillaSGD(params_list, 0.05)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=net.batch_size, shuffle=True) 

# Training loop
for epochs in range(20):

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):


        gradient_updater.zero_grad() # reset previous gradients

        soft_out = net(Xbatch)
        loss = ft.crossEntropy_MNIST(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
        loss.backward() # update gradient
        gradient_updater.step() # update weights
        
        if i % 30 == 0:     
            print(f"batch: {i}")

        # first inference

    pred = net(ft.Tensor(Xtest[:100])).argmax(axis=-1)
    accuracy = (pred.data[:100] == Ytest[:100]).mean()
    print(f"test accuracy: {accuracy}")

print(f"epoch {epochs}")


# first inference
pred = net(ft.Tensor(Xtest[:1000])).argmax(axis=-1)
accuracy = (pred.data[:1000] == Ytest[:1000]).mean()
print(f"test accuracy: {accuracy}")

# save weights
ft.save("checkpoints_weights/CIFAR_CNN.npz", params_list, overwrite=True)

# 2nd inference
net2 = OptiCnnNet()
params_list2 = net2.parameters()

# load weights used in 1st inference by net into net2
ft.load("checkpoints_weights/CIFAR_CNN.npz", params_list2)


pred = net2(ft.Tensor(Xtest[:1000])).argmax(axis=-1)
accuracy = (pred.data[:1000] == Ytest[:1000]).mean()

print(f"test accuracy: {accuracy}")





