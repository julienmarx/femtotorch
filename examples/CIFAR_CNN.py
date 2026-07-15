
import femtotorch as ft
import numpy as np


class MaxPoolCnnNet:
    """
    65 %
    """
    def __init__(self):
        self.batch_size = 64 

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
 

# Initialization
Xtrain, Ytrain, Xtest, Ytest = ft.load_cifar10("data/cifar10")
net = MaxPoolCnnNet()
params_list = net.parameters()
gradient_updater = ft.VanillaSGD(params_list, 0.05)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=net.batch_size, shuffle=True) 

# Training loop
for epochs in range(20):

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):


        gradient_updater.zero_grad() # reset previous gradients

        soft_out = net(Xbatch)
        loss = ft.cross_entropy(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
        loss.backward() # update gradient
        gradient_updater.step() # update weights
        
        if i % 30 == 0:     
            print(f"batch: {i}")

        # first inference

    with ft.no_grad():
        pred = net(ft.Tensor(Xtest[:10000])).argmax(axis=-1)
    accuracy = (pred.data[:10000] == Ytest[:10000]).mean()
    print(f"epoch{epochs} ,test accuracy: {accuracy}")

    # checkpoint each epoch so a crash mid-run doesn't lose progress
    ft.save("checkpoints_weights/CIFAR_CNN.npz", params_list, overwrite=False)

print(f"epoch {epochs}")

# first inference
with ft.no_grad():
    pred = net(ft.Tensor(Xtest[:1000])).argmax(axis=-1)
accuracy = (pred.data[:1000] == Ytest[:1000]).mean()
print(f"test accuracy: {accuracy}")

# save weights
ft.save("checkpoints_weights/CIFAR_CNN.npz", params_list, overwrite=False)

# 2nd inference
net2 = MaxPoolCnnNet()
params_list2 = net2.parameters()

# load weights used in 1st inference by net into net2
ft.load("checkpoints_weights/CIFAR_CNN.npz", params_list2)


with ft.no_grad():
    pred = net2(ft.Tensor(Xtest)).argmax(axis=-1)
accuracy = (pred.data == Ytest).mean()

print(f"test accuracy: {accuracy}")





