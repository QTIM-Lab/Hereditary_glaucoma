"""Defines model, training and data loading for glaucoma detection.
Taken from https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
"""

import os
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from glaucoma_fl.data.chaksu import ChaksuDataset

# model
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)

    def forward(self, x):
        return self.resnet18(x)

# data
def load_data(image_folders: dict, label_files: dict, partition: str, batch_size: int):
    """Load Chaksu dataset."""

    image_folder_train: str = os.path.join(image_folders['train'], partition)
    label_file_train: str = label_files['train'].format(partition)
    image_folder_test: str = os.path.join(image_folders['test'], partition)
    label_file_test: str = label_files['test'].format(partition)

    # data augmentation - ResNet18 default data augmentation
    transform = ResNet18_Weights.DEFAULT.transforms()

    # create dataset instance
    train_dataset = ChaksuDataset(image_folder_train, label_file_train, transform=transform)
    test_dataset = ChaksuDataset(image_folder_test, label_file_test, transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    return trainloader, testloader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # extension - these hyperparameters could come from configuration
    net.train()
    running_loss = 0.0
    for epoch_n in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"].float()
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return {'loss': avg_trainloss}

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = 0.0
    preds = []
    all_labels = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].float().to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    all_labels = np.concatenate(all_labels)
    # compute metrics
    preds_binary = (preds > 0.5).astype(int)
    accuracy = (preds_binary == all_labels).mean()

    auc_score = roc_auc_score(all_labels, preds)
    return {'loss': loss, 'accuracy': accuracy, 'auc_score': auc_score}

def get_weights(net):
    """Collect the weights from the model to send them to the server."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Given weights sent by the server, set the weights in the model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
