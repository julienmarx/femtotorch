
import femtotorch as ft
import numpy as np
import time


BATCH_SIZE = 64
class VGG_BN(ft.Module):
    """
    Reliably reach 80 to 81 %
    """
    def __init__(self):

        self.conv1 = ft.OptiConv2d(in_channels=3, out_channels=64, kernel_size=3, stride =1, padding=1, bias=False) 
        self.batchnorm1 = ft.BatchNorm2d(num_features=64)
        self.conv1b = ft.OptiConv2d(in_channels=64, out_channels=64, kernel_size=3, stride =1, padding=1, bias=False)
        self.batchnorm1b = ft.BatchNorm2d(num_features=64)
        self.pool1 = ft.MaxPool2d(kernel_size=2)


        self.conv2 = ft.OptiConv2d(in_channels=64, out_channels=128, kernel_size=3, stride =1, padding=1, bias=False) 
        self.batchnorm2 = ft.BatchNorm2d(num_features=128)
        self.conv2b = ft.OptiConv2d(in_channels=128, out_channels=128, kernel_size=3, stride =1, padding=1, bias=False)
        self.batchnorm2b = ft.BatchNorm2d(num_features=128)
        self.pool2 = ft.MaxPool2d(kernel_size=2)


        self.conv3 = ft.OptiConv2d(in_channels=128, out_channels=256, kernel_size=3, stride =1, padding=1, bias=False) 
        self.batchnorm3 = ft.BatchNorm2d(num_features=256)
        self.conv3b = ft.OptiConv2d(in_channels=256, out_channels=256, kernel_size=3, stride =1, padding=1, bias=False)
        self.batchnorm3b = ft.BatchNorm2d(num_features=256)
        self.pool3 = ft.MaxPool2d(kernel_size=2)

        self.out_conv3 = self.conv3.size_map(8, 8)
        self.out_pool3 = (self.out_conv3 // 4)                
        self.model = ft.MLP(self.out_pool3, [512, 10]) 

    def forward(self, X):

        x = self.conv1(X.reshape(-1, 3, 32, 32)) # the -1 allows flexibility on the last batch 
        x = self.batchnorm1(x).relu()
        x = self.conv1b(x)
        x = self.batchnorm1b(x).relu()
        x = self.pool1(x)

        x = self.conv2(x) # the -1 allows flexibility on the last batch 
        x = self.batchnorm2(x).relu()
        x = self.conv2b(x)
        x = self.batchnorm2b(x).relu()
        x = self.pool2(x)

        x = self.conv3(x) # the -1 allows flexibility on the last batch 
        x = self.batchnorm3(x).relu()
        x = self.conv3b(x)
        x = self.batchnorm3b(x).relu()
        x = self.pool3(x)

        x = x.reshape(-1, self.out_pool3)

        logits = self.model(x)

        return logits
    
    def set__batchnorm(self, training = True):
        self.batchnorm1.set_training(training)
        self.batchnorm1b.set_training(training)

        self.batchnorm2.set_training(training)
        self.batchnorm2b.set_training(training)

        self.batchnorm3.set_training(training)
        self.batchnorm3b.set_training(training)
    
        
    
    def parameters(self):
        return [*self.model.parameters(),
                 *self.conv1.parameters(), *self.conv2.parameters(), *self.conv3.parameters(),
                 *self.conv1b.parameters(), *self.conv2b.parameters(), *self.conv3b.parameters(),
                 *self.batchnorm1.parameters(), *self.batchnorm2.parameters(), *self.batchnorm3.parameters(),
                 *self.batchnorm1b.parameters(), *self.batchnorm2b.parameters(), *self.batchnorm3b.parameters()
                 ]
    
 

# Initialization
Xtrain, Ytrain, Xtest, Ytest = ft.load_cifar10("data/cifar10")
net = VGG_BN()
params_list = net.parameters()
gradient_updater = ft.SGD_Moment(params_list, 0.05)
lr_scheduler = ft.CosineScheduler(gradient_updater, 30)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=BATCH_SIZE, shuffle=True) 

# Training loop
for epochs in range(1):

    t1 = time.perf_counter()

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):
        
        ft.synchronize()
        t0 = time.perf_counter()

        gradient_updater.zero_grad() # reset previous gradients

        raw_logits = net(Xbatch)
        loss = ft.softmax_cross_entropy(raw_logits, Ybatch).mean()
        loss.backward() # update gradient
        gradient_updater.step() # update weights
        
        if i % 30 == 0:     
            print(f"batch: {i}, \n loss: {loss.data}")
            break

            # memory consumption on gpu
            stats = ft.memory_stats()
            if stats is not None:
                print(f"pool {stats['pool_used']/1e6:.0f} / {stats['pool_total']/1e6:.0f} MB, "
                    f"device free {stats['dev_free']/1e6:.0f} MB, \n")
            

        # first inference
    # lr decay per epoch
    lr_scheduler.step()
    print(f"learning_rate:{gradient_updater.get_lr()}")
    profiler = ft.Profiler()
    profiler.profile_model(net, Xbatch, Ybatch)   # X: one input batch, y: the class indices


    # test per epoch
    with ft.no_grad():
        net.set__batchnorm(training=False)

        correct = 0
        for start in range(0, 10000, 500):
            pred = net(ft.Tensor(Xtest[start:start+500])).argmax(axis=-1)
            correct += (ft.to_cpu(pred.data) == Ytest[start:start+500]).sum()
            

    accuracy = correct / 1000
    print(f"epoch{epochs} ,test accuracy: {accuracy}")
    net.set__batchnorm(training= True)







