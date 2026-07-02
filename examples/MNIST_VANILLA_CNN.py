import femtotorch as ft
import numpy as np

batch_size = 64

Xtrain, Ytrain, Xtest, Ytest = ft.load_mnist("data/fashion_mnist")
Xtrain, Ytrain = Xtrain[:10000], Ytrain[:10000]
conv = ft.Vanilla_Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride =2, padding=1)
model = ft.MLP(784, [256, 10])
gradient_updater = ft.VanillaSGD([*model.parameters(), *conv.parameters()], 0.05)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, batch_size=batch_size, shuffle=True) 




for epochs in range(1):

    for i, (Xbatch, Ybatch) in enumerate(batch_generator):

        gradient_updater.zero_grad() # reset previous gradients
        x = conv(Xbatch.reshape(-1, 1, 28, 28)).relu() # the -1 allows flexibility on the last batch 
        x = x.reshape(-1, 4*14*14)
        soft_out = ft.softmax(model(x)) # compute forward pass
        loss = ft.crossEntropy_MNIST(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
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





