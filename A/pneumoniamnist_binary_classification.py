from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import sklearn
import medmnist
from medmnist import INFO, Evaluator
from torchvision.transforms import ToTensor
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score

DataClass = medmnist.PneumoniaMNIST

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
]) # Convert all data points between range -1 to 1

#Setup training and test data

train_data = DataClass(split='train', transform=data_transform, download=True, root="./../Datasets/PneumoniaMNIST", as_rgb= True)

test_data = DataClass(split='test', transform=data_transform, download=True, root="./../Datasets/PneumoniaMNIST", as_rgb= True)

train_metrix, train_label = train_data[0] 
print(train_metrix)
print(train_metrix.shape)
print(train_label)