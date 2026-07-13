import femtotorch as ft
import numpy as np

Xtrain, Ytrain, Xtest, Ytest = ft.load_mnist("data/fashion_mnist")


model = ft.MLP(784, [256, 10])
gradient_updater = ft.VanillaSGD(model.parameters(), 0.05)
batch_generator =  ft.Dataloader(Xtrain, Ytrain, 32, shuffle=True) 

for epochs in range(20):

    for Xbatch, Ybatch in batch_generator:

        gradient_updater.zero_grad() # reset previous gradients
        soft_out = ft.softmax(model(Xbatch)) # compute forward pass
        loss = ft.cross_entropy(soft_out, ft.one_hot(Ybatch)).mean() # loss function take mean of the loss of all vectors in the batch
        loss.backward() # update gradient
        gradient_updater.step() # update weights

        # inference

    pred = ft.softmax(model(ft.Tensor(Xtest))).argmax(axis = -1)
    accuracy = (pred.data == Ytest).mean()
    

    pred2 = ft.softmax(model(ft.Tensor(Xtrain))).argmax(axis = -1)
    accuracy2 = (pred2.data == Ytrain).mean()
    print(accuracy, f"train:{accuracy2}")