from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms

import torchvision

device = torch.device("cuda:0") # Let's make sure GPU is available!

tfs = transforms.Compose([
    transforms.RandomRotation(50, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
trainset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=tfs)
testset = torchvision.datasets.MNIST('/files/', train=False, download=True, transform=tfs)
train_labeled_set, train_unlabeled_set = torch.utils.data.random_split(trainset, [3000, 57000])

train_labeled_loader = torch.utils.data.DataLoader(train_labeled_set, batch_size=64, shuffle=True)
train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64)

class Flattener(nn.Module):
    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)

nn_model = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1152, 10)
          )

nn_model.type(torch.cuda.FloatTensor)
nn_model.to(device)

loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
optimizer = optim.SGD(nn_model.parameters(), lr=1e-1, weight_decay=1e-4)


def compute_accuracy(model, loader):
  val_acc = 0.0, 0.0
  for img, label in loader:
    img_gpu = img.to(device)
    label_gpu = label.to(device)
    with torch.no_grad():
      label_hat = model(img_gpu)
        
    val_acc += (label_hat.argmax(axis=-1) == label_gpu).type(torch.float32).mean().item()
  val_acc /= len(loader)
  return val_acc


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    
    loss_history = []
    train_history = []
    val_history = []

    errors_num = 0
    epsilon = 0.01
    max_num = 4
  
    print("Unlabeled data")  

    for epoch in range(num_epochs):
        model.train()
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        prev_loss = 0.0
        for i_step, (x, y) in enumerate(train_loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)    
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]
            
            loss_accum += loss_value

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples

        if(abs(ave_loss - prev_loss) <= epsilon):
          errors_num += 1
        else:
          errors_num = 0
        if(errors_num > max_num): 
          break
        prev_loss = ave_loss

        val_accuracy = compute_accuracy(model, val_loader)
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))
        
    return loss_history, train_history, val_history
        
loss_history, train_history, val_history = train_model(nn_model, train_labeled_loader, test_loader, loss, optimizer, 5)



def train1_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    
    loss_history = []
    train_history = []
    val_history = []

    errors_num = 0
    epsilon = 0.01
    max_num = 4
    alpha = 3.1

    print("Unlabeled data")    

    for epoch in range(num_epochs):
        model.train()
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        prev_loss = 0.0
        steps = 100
        for i_step, (x, y) in enumerate(train_loader):
            alpha = alpha*(steps-100)/(7500-100)
            x_gpu = x.to(device)
            prediction = model(x_gpu)    
            pred = prediction.data.max(1)[1]
            target1 = pred
            target1 = target1.view(target1.size()[0])

            optimizer.zero_grad()
            output1 = model(x_gpu)
            loss_value = alpha*loss(output1, target1)
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == target1)
            total_samples += target1.shape[0]

            steps += 1         
            
            loss_accum += loss_value

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples

        if(abs(ave_loss - prev_loss) <= epsilon):
          errors_num += 1
        else:
          errors_num = 0
        if(errors_num > max_num): 
          break
        prev_loss = ave_loss

        val_accuracy = compute_accuracy(model, val_loader)
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))
        
    return loss_history, train_history, val_history

loss_history, train_history, val_history = train1_model(nn_model, train_unlabeled_loader, test_loader, loss, optimizer, 5)