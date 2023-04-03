import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader
WORK_PART = 2
if WORK_PART == 1:
    """
    Задачние 1 Dataset moons 
    """
    sns.set(style="darkgrid", font_scale=1.4)

    X, y = make_moons(n_samples=10000, random_state=42, noise=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

    X_train_t = torch.FloatTensor(X_train)# YOUR CODE GOES HERE
    y_train_t = torch.FloatTensor(y_train)# YOUR CODE GOES HERE
    X_val_t = torch.FloatTensor(X_val) # YOUR CODE GOES HERE
    y_val_t =  torch.FloatTensor(y_val)# YOUR CODE GOES HERE

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_dataloader = DataLoader(train_dataset, batch_size=128, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, drop_last=True)


    class LinearRegression(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(in_features))
            self.bias = bias
            if bias:
                self.bias_term = nn.Parameter(torch.randn(1))  # YOUR CODE GOES HERE

        def forward(self, x):
            x = x @ self.weights  # YOUR CODE GOES HERE


            if self.bias:
                x += self.bias_term  # YOUR CODE GOES HERE

            return x

    linear_regression = LinearRegression(2, 1)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(linear_regression.parameters(), lr=0.05)

    tol = 1e-3
    losses = []
    max_epochs = 100
    prev_weights = torch.zeros_like(linear_regression.weights)
    stop_it = False
    for epoch in range(max_epochs):
        for it, (X_batch, y_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outp = linear_regression(X_batch)  # YOUR CODE. Use linear_regression to get outputs

            loss = loss_function(outp, y_batch)  # YOUR CODE. Compute loss
            loss.backward()
            losses.append(loss.detach().flatten()[0])
            optimizer.step()
            probabilities = torch.sigmoid(outp)  # YOUR CODE. Compute probabilities
            preds = (probabilities > 0.5).type(torch.long)
            batch_acc = (preds.flatten() == y_batch).type(torch.float32).sum() / y_batch.size(0)
            if (it + epoch * len(train_dataloader)) % 100 == 0:
                print(f"Iteration: {it + epoch * len(train_dataloader)}\nBatch accuracy: {batch_acc}")
            current_weights = linear_regression.weights.detach().clone()
            if (prev_weights - current_weights).abs().max() < tol:
                print(f"\nIteration: {it + epoch * len(train_dataloader)}.Convergence. Stopping iterations.")
                stop_it = True
                break
            prev_weights = current_weights
        if stop_it:
            break

    @torch.no_grad()
    def predict(dataloader, model):
        model.eval()
        predictions = np.array([])
        for x_batch, _ in dataloader:
            probabilities = torch.sigmoid(model(x_batch))#YOUR CODE. Compute predictions
            preds = (probabilities > 0.5).type(torch.long)
            predictions = np.hstack((predictions, preds.numpy().flatten()))
        return predictions.flatten()

    preds = predict(val_dataloader, linear_regression)
    print(preds)

    from sklearn.metrics import accuracy_score
    y_val_drop_last = y_val[: y_val.shape[0] - (y_val.shape[0] % 128 )]

    acc = accuracy_score(y_val_drop_last, preds)
    print(acc)

if WORK_PART == 2:
    """
    Заданиче 2.1 
    Датасет MNIST
    """
    import os
    from torchvision.datasets import MNIST
    from torchvision import transforms as tfs

    data_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5), (0.5))
    ])

    # install for train and test
    root = './'
    train_dataset = MNIST(root, train=True, transform=data_tfs, download=True)
    val_dataset = MNIST(root, train=False, transform=data_tfs, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=128, drop_last=True)  # YOUR CODE GOES HERE
    valid_dataloader = DataLoader(val_dataset, batch_size=128, drop_last=True)  # YOUR CODE GOES HERE

    activation = nn.ELU

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=128),
        activation(),
        nn.Linear(in_features=128, out_features=10)

        # YOUR CODE. Add layers to your sequential class
    )
    criterion = nn.CrossEntropyLoss() # YOUR CODE. Select a loss function
    optimizer = torch.optim.Adam(model.parameters())
    loaders = {"train": train_dataloader, "valid": valid_dataloader}
    it = 0
    max_epochs = 10
    accuracy = {"train": [], "valid": []}
    for epoch in range(max_epochs):
        for k, dataloader in loaders.items():
            it += 1
            epoch_correct = 0
            epoch_all = 0
            for x_batch, y_batch in dataloader:
                if k == "train":
                # YOUR CODE. Set model to ``train`` mode and calculate outputs. Don't forget zero_grad!
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)

                else:
                # YOUR CODE. Set model to ``eval`` mode and calculate outputs
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                preds = outp.argmax(-1)
                correct = (preds == y_batch).sum() # YOUR CODE GOES HERE
                all = y_batch.shape[0] # YOUR CODE GOES HERE
                epoch_correct += correct.item()
                epoch_all += all
                if k == "train":
                    loss = criterion(outp, y_batch)
                    # YOUR CODE. Calculate gradients and make a step of your optimizer
                    # делаем шаг градиентного спуска
                    loss.backward()
                    optimizer.step()

            if k == "train":
                print(f"Epoch: {epoch + 1}")
            print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")
            print(it)
            accuracy[k].append(epoch_correct / epoch_all)
            print(len(accuracy[k]))

    elu_accuracy = accuracy["valid"]

