import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

from utils import DistillationLoss
import matplotlib.pyplot as plt
import numpy as np

def get_transforms(params, train=True):
    mean, std = params["mean"], params["std"]

    if params["dataset"] == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def get_loaders(params):
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)


    datasetler = train_val_dataset(train_ds)
    print(len(datasetler['train']))
    print(len(datasetler['val']))
    # The original dataset is available in the Subset class
    print(datasetler['train'].dataset)
    train_ds = datasetler['train']
    val_ds = datasetler['val']


    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader

def label_smoothing(labels, num_classes, sm_factor = 0.1 ):
    """
    label smoothing applied to one-hot encoded vector
    arXiv:2003.02819 
    """
    new_labels = (1-sm_factor)*labels + sm_factor/num_classes
    return new_labels

def train_one_epoch(tmodel, model, loader, optimizer, criterion, device, log_interval, nc, kd='soft'):
    model.train()
    tmodel.eval()
    
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):  
        imgs, labels = imgs.to(device), labels.to(device)
        if kd =='soft':
            tlogits = tmodel(imgs)

        optimizer.zero_grad()
        slogits  = model(imgs)

        loss = criterion(slogits, tlogits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += slogits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def run_training(tmodel, model, params, device):
    train_loader, val_loader = get_loaders(params)
    #knowledge distillation
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    criterion_val = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = None

    Loss_train = []
    Loss_val = []
    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(tmodel, model, train_loader, optimizer,
                                          criterion, device, params["log_interval"], params['num_classes'], kd='soft')
        val_loss, val_acc = validate(model, val_loader, criterion_val, device)
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  Saved best model (val_acc={best_acc:.4f})")

        #for visualization of train/val loss graph
        Loss_train.append(tr_loss)
        Loss_val.append(val_loss)

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    #visualize
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.plot(np.arange(params["epochs"]), Loss_train, color = 'blue', label = 'Train')
    plt.plot(np.arange(params["epochs"]), Loss_val, color = 'red', label = 'Validation')
    plt.legend()
    plt.show()