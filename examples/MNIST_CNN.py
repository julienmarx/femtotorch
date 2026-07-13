"""
Version to compare with vanilla cnn

import femtotorch as ft
import numpy as np

batch_size = 64

Xtrain, Ytrain, Xtest, Ytest = ft.load_mnist("data/fashion_mnist")
Xtrain, Ytrain = Xtrain[:10000], Ytrain[:10000]
conv = ft.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride =2, padding=1)
model = ft.MLP(784, [256, 10])
gradient_updater = ft.VanillaSGD([*model.parameters(), *conv.parameters()], 0.05)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=batch_size, shuffle=True) 




for epochs in range(1):

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):

        gradient_updater.zero_grad() # reset previous gradients
        x = conv(Xbatch.reshape(-1, 1, 28, 28)).relu() # the -1 allows flexibility on the last batch 
        x = x.reshape(-1, 4*14*14)
        soft_out = ft.softmax(model(x)) # compute forward pass
        loss = ft.cross_entropy(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
        loss.backward() # update gradient
        gradient_updater.step() # update weights

        print(f"batch {i}")

        if i % 30 == 0:
            pred = ft.softmax(model((conv(ft.Tensor(Xtest[:100]).reshape(100, 1, 28, 28)).relu()).reshape(100, 784))).argmax(axis = -1)
            accuracy = (pred.data == Ytest[:100]).mean()
            print(f"test accuracy: {accuracy}")
    # inference

    pred = ft.softmax(model((conv(ft.Tensor(Xtest[:2000]).reshape(-1, 1, 28, 28)).relu()).reshape(-1, 784))).argmax(axis = -1)
    accuracy = (pred.data == Ytest[:2000]).mean()


    #pred2 = ft.softmax(model(ft.Tensor(Xtrain))).argmax(axis = -1)
    #accuracy2 = (pred2.data == Ytrain).mean()
    #print(accuracy, f"train:{accuracy2}")

    print(f"test accuracy: {accuracy}")





"""

import femtotorch as ft
import numpy as np

batch_size = 64

Xtrain, Ytrain, Xtest, Ytest = ft.load_mnist("data/fashion_mnist")

conv1 = ft.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride =1, padding=1) 
out_conv1 = conv1.size_map(28, 28)

conv2 = ft.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride = 2, padding=1)
out_conv2 = conv2.size_map(28, 28) # since conv1 output is 32 * 28 * 28

conv3 = ft.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride =1, padding=1) 
out_conv3 = conv3.size_map(14, 14) # since conv2 output is 32 * 14 * 14

conv4 = ft.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride = 2, padding=1)
out_conv4 = conv4.size_map(14, 14) # since conv3 output is 64 * 14 * 14

model = ft.MLP(out_conv4, [256, 10]) 
gradient_updater = ft.VanillaSGD([*model.parameters(), *conv1.parameters(), *conv2.parameters(), *conv3.parameters(), *conv4.parameters()], 0.05)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=batch_size, shuffle=True) 




for epochs in range(20):

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):

        gradient_updater.zero_grad() # reset previous gradients

        x = conv1(Xbatch.reshape(-1, 1, 28, 28)).relu() # the -1 allows flexibility on the last batch 

        x = conv2(x).relu()

        x = conv3(x).relu()

        x = conv4(x).relu()
        x = x.reshape(-1, out_conv4)

        soft_out = ft.softmax(model(x)) # compute forward pass
        loss = ft.cross_entropy(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
        loss.backward() # update gradient
        gradient_updater.step() # update weights
        
        if i % 30 == 0:
            print(f"batch: {i}")

    print(f"epoch {epochs}")

    # inference
    x = conv1(ft.Tensor(Xtest[:1000]).reshape(-1, 1, 28, 28)).relu()
    x = conv2(x).relu()
    x = conv3(x).relu()
    x = conv4(x).relu()
    x = x.reshape(-1, out_conv4)
    pred = ft.softmax(model(x)).argmax(axis=-1)
    

    accuracy = (pred.data[:1000] == Ytest[:1000]).mean()

    print(f"test accuracy: {accuracy}")





