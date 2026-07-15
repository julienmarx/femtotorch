
import femtotorch as ft
import numpy as np


class VggNet:
    """
    first model to reach 80%
    """
    def __init__(self):
        self.batch_size = 64 

        self.conv1 = ft.OptiConv2d(in_channels=3, out_channels=32, kernel_size=3, stride =1, padding=1, bias=False) 
        self.batchnorm1 = ft.BatchNorm2d(num_features=32)
        self.pool1 = ft.MaxPool2d(kernel_size=2)

        self.conv2 = ft.OptiConv2d(in_channels = 32, out_channels=64, kernel_size=3, stride = 1, padding=1, bias=False)
        self.batchnorm2 = ft.BatchNorm2d(num_features=64)
        self.pool2 = ft.MaxPool2d(kernel_size=2)

        self.conv3 = ft.OptiConv2d(in_channels=64, out_channels=128, kernel_size=3, stride =1, padding=1, bias=False) 
        self.out_conv3 = self.conv3.size_map(8, 8) # since pool2 output is 32 * 8 * 8
        self.batchnorm3 = ft.BatchNorm2d(num_features=128)
        self.pool3 = ft.MaxPool2d(kernel_size=2)
        self.out_pool3 = (self.out_conv3 // 4)

        self.model = ft.MLP(self.out_pool3, [128, 10]) 

    def __call__(self, X):

        x = self.conv1(X.reshape(-1, 3, 32, 32)).relu() # the -1 allows flexibility on the last batch 
        x = self.batchnorm1(x)
        x = self.pool1(x)

        x = self.conv2(x).relu()
        x = self.batchnorm2(x)
        x = self.pool2(x)

        x = self.conv3(x).relu()
        x = self.batchnorm3(x)
        x = self.pool3(x)

        x = x.reshape(-1, self.out_pool3)

        soft_out = ft.softmax(self.model(x))

        return soft_out
    
    def set__batchnorm(self, training = True):
        self.batchnorm1.set_training(training)
        self.batchnorm2.set_training(training)
        self.batchnorm3.set_training(training)
    
        
    
    def parameters(self):
        return [*self.model.parameters(), *self.conv1.parameters(), *self.conv2.parameters(), *self.conv3.parameters(), *self.batchnorm1.parameters(), *self.batchnorm2.parameters(), *self.batchnorm3.parameters()]
    
 

# Initialization
Xtrain, Ytrain, Xtest, Ytest = ft.load_cifar10("data/cifar10")
net = VggNet()
params_list = net.parameters()
gradient_updater = ft.SGD_Moment(params_list, 0.05)
lr_scheduler = ft.CosineScheduler(gradient_updater, 30)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=net.batch_size, shuffle=True) 

# Training loop
for epochs in range(30):

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):


        gradient_updater.zero_grad() # reset previous gradients

        soft_out = net(Xbatch)
        loss = ft.cross_entropy(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
        loss.backward() # update gradient
        gradient_updater.step() # update weights
        
        if i % 50 == 0:     
            print(f"batch: {i}, \n loss: {loss.data}")

        # first inference
    # lr decay per epoch
    lr_scheduler.step()
    print(f"learning_rate:{gradient_updater.get_lr()}")

    # test per epoch
    with ft.no_grad():
        net.set__batchnorm(training=False)
        pred = net(ft.Tensor(Xtest[:10000])).argmax(axis=-1)
    accuracy = (ft.to_cpu(pred.data[:10000]) == Ytest[:10000]).mean()
    print(f"epoch{epochs} ,test accuracy: {accuracy}")
    net.set__batchnorm(training= True)
    







