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

data_flag = 'PneumoniaMNIST'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

#info = INFO[data_flag]
#task = info['task']
#n_channels = info['n_channels']
#n_classes = len(info['label'])
DataClass = medmnist.PneumoniaMNIST

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
]) # Convert all data points between range -1 to 1

#Setup training and test data

train_data = DataClass(split='train', transform=data_transform, download=True, root="./../Datasets/PneumoniaMNIST")

test_data = DataClass(split='test', transform=data_transform, download=True, root="./../Datasets/PneumoniaMNIST")


with np.load('./../Datasets/PneumoniaMNIST/pneumoniamnist.npz') as data:
    train_images = data['train_images']
    train_labels = data['train_labels']

    val_images = data['val_images']
    val_labels = data['val_labels']

    test_images = data['test_images']
    test_labels = data['test_labels']

train_images = stats.zscore(train_images, axis=None)
train_images = train_images.reshape(train_images.shape[0], -1)
print(np.shape(train_images))
train_labels = train_labels.ravel()
print(train_labels)
print(np.shape(train_labels))
svm_model = svm.SVC(kernel='linear', C = 50.0)
svm_model.fit(train_images, train_labels)

test_images = test_images.reshape(test_images.shape[0], -1)

predict_values = svm_model.predict(test_images)

print('===== Accuracy Score ===== ')
print(accuracy_score(test_labels.ravel(), predict_values) * 100 )
image, label = train_data[0]

#print(type(image))
#print(type(label))

#print(image.numpy())
#print(image.numpy().shape)
clf = svm.SVC(kernel='linear', C = 50.0)

#clf.fit()

print(type(train_data))


print(train_data.montage(length=1))
