import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.FrankNet import FrankNet

device = "cuda" if torch.cuda.is_available() else "cpu"

#### Load CIFAR10 dataset
transform = Compose([
    Resize((227,227)),
    ToTensor(),
    Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
])

training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

batch_size = 8

train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


#### Training and Testing functions
def train(dataloader, model, loss_fn, optimizer):

    # size of the dataset
    size = len(dataloader.dataset)

    # model in the training process
    model.train()

    totalLoss = 0

    for batch, (X, y) in enumerate(dataloader):
        # transforms the inputs to the device format (CPU or GPU)
        X, y = X.to(device), y.to(device)

        # prediction
        pred = model(X)

        # function loss estimation
        loss = loss_fn(pred, y)
        totalLoss += loss

        #### Backpropagation

        # gradient clearing
        optimizer.zero_grad()

        # gradient estimation
        loss.backward()

        # optimization of the parameters
        optimizer.step()

    print(f"Epoch average loss: {totalLoss/len(dataloader):>7f}")

def test(dataloader, model, loss_fn, batch_size, epoch, lr, optimizer):

    # size of the dataset and number of batches
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # model in the testing process
    model.eval()

    test_loss, correct = 0, 0

    # initialize lists of predictions and labels for confusion matrix
    all_preds, all_labels = [], []

    # disable gradient calculation
    with torch.no_grad():

        for X, y in dataloader:
            # transform the inputs to the device format (CPU or GPU)
            X, y = X.to(device), y.to(device)

            # prediction
            pred = model(X)

            # get the class with the highest probability as the prediction and append to the list
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # calculate loss
            test_loss += loss_fn(pred, y).item()
            
            # calculate accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # average loss and accuracy
    test_loss /= num_batches
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # save results to file
    try:
        with open('frank_results/frank_' + str(batch_size) + '_' + str(epoch) + '_' + str(lr) + '_' + str(optimizer) + '.txt', 'w') as f:
            f.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    except:
        print("Error writing to file")


    #### Confusion Matrix
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # save confusion matrix as image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig('confusion_matrix/frank_cm_' + str(batch_size) + '_' + str(epoch) + '_' + str(lr) + '_' + str(optimizer) + '.png')


#### Model
franknet = FrankNet().to(device)
print(franknet)

#### Hyperparameters(batch_size is already defined at the beginning of the file)
learning_rate = 0.01
epochs = 30
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(franknet.parameters(), lr=learning_rate)


#### Training
start, end = 0, 0
for t in range(epochs):
    print(f"-------------------------------\nEpoch {t+1}")
    start = time.time()
    train(train_loader, franknet, loss_fn, optimizer)
    end = time.time()
    print(f"Epoch time: {end-start}")

#### Testing
start = time.time()
test(test_loader, franknet, loss_fn, batch_size, epochs, learning_rate, "sgd")
end = time.time()
print(f"Testing time: {end-start}")
print("\nDone!")
