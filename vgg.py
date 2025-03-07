import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.VGG13 import VGG13

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def train(dataloader, model, loss_fn, optimizer):
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)
    # Indica que o modelo está em processo de treinamento
    model.train()

    # Define a loss total do treinamento
    totalLoss = 0

    # Itera sobre os lotes
    for batch, (X, y) in enumerate(dataloader):
        # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
        X, y = X.to(device), y.to(device)

        # Faz a predição para os valores atuais dos parâmetros
        pred = model(X)

        # Estima o valor da função de perda
        loss = loss_fn(pred, y)

        # Incrementa a loss total
        totalLoss += loss

        # Backpropagation

        # Limpa os gradientes
        optimizer.zero_grad()

        # Estima os gradientes
        loss.backward()

        # Atualiza os pesos da rede
        optimizer.step()

        if batch % 128 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    print(f"Epoch average loss: {totalLoss/len(dataloader):>7f}")

def test(dataloader, model, loss_fn, batch_size, epoch, lr, optimizer):
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)

    # Obtém o número de lotes (iterações)
    num_batches = len(dataloader)

    # Indica que o modelo está em processo de teste
    model.eval()

    # Inicializa a perda de teste e a quantidade de acertos com 0
    test_loss, correct = 0, 0

    # Inicializa as listas de predições e rótulos
    all_preds, all_labels = [], []

    # Desabilita o cálculo do gradiente
    with torch.no_grad():
        # Itera sobre o conjunto de teste
        for X, y in dataloader:
            # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
            X, y = X.to(device), y.to(device)
            # Realiza a predição
            pred = model(X)

            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            

            # Calcula a perda
            test_loss += loss_fn(pred, y).item()
            # Verifica se a predição foi correta
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Determina a perda média e a proporção de acertos
    test_loss /= num_batches
    correct /= size
    
    # mostra a acurácia e a perda
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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

    plt.savefig('vgg_cm_' + batch_size + '_' + epoch + '_' + lr + '_' + optimizer + '.png')

vgg13 = VGG13().to(device)
print(vgg13)

learning_rate = 0.001
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg13.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"-------------------------------\nEpoch {t+1}")
    train(train_loader, vgg13, loss_fn, optimizer)
test(test_loader, vgg13, loss_fn, batch_size, epochs, learning_rate, 'adam')
print("Done!")