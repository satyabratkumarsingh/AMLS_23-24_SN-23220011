from tqdm import tqdm
import numpy as np

import medmnist
from medmnist import INFO, Evaluator
from utils.base_logger import get_logger
import torchvision.transforms as transforms
from B.convolutional_neural_network import ConvolutionalNeuralNetwork
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data

class MultiClassification:
    def __init__(self):
        self.logger = get_logger()
        self.data_class = medmnist.PathMNIST
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.download_folder = "./Datasets/PathMNIST"
        self.BATCH_SIZE = 128

    def download_images(self):
        self.logger.info('Downloading images ..... ')
        train_dataset = self.data_class(split='train', transform=self.data_transform, download=True, root = self.download_folder, as_rgb= False)
        test_dataset = self.data_class(split='test', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= False)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        train_metrix, train_label = train_dataset[0]
        self.logger.info(train_metrix)
        self.logger.info(train_metrix.shape)

    def train_model(self):

        train_dataset = self.data_class(split='train', transform=self.data_transform, download=True, root = self.download_folder, as_rgb= True)
        test_dataset = self.data_class(split='test', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= True)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        train_metrix, train_label = train_dataset[0]


        learning_rate = 0.001
        cnn = ConvolutionalNeuralNetwork()
        criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)

        num_epochs = 10

        cnn.train()
            
        # Train the model
    #     total_step = len(loaders['train'])
            
    #     for epoch in range(num_epochs):
    #         for i, (images, labels) in enumerate(loaders['train']):
                
    #             # gives batch data, normalize x when iterate train_loader
    #             b_x = Variable(images)   # batch x
    #             b_y = Variable(labels)   # batch y
    #             output = cnn(b_x)[0]               
    #             loss = loss_func(output, b_y)
                
    #             # clear gradients for this training step   
    #             optimizer.zero_grad()           
                
    #             # backpropagation, compute gradients 
    #             loss.backward()    
    #             # apply gradients             
    #             optimizer.step()                
                
    #             if (i+1) % 100 == 0:
    #                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    #                     .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    #             pass
            
    #         pass
        
    
    # pass



    #     NUM_EPOCHS = 3

    #     for epoch in range(NUM_EPOCHS):
    #         train_correct = 0
    #         train_total = 0
    #         test_correct = 0
    #         test_total = 0
    
    #         model.train()

    #     for inputs, targets in tqdm(train_loader):

    #         print(inputs)
    #     # forward + backward + optimize
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
            
    #         targets = targets.squeeze().long()
    #         loss = criterion(outputs, targets)
               
            
    #     loss.backward()
    #     optimizer.step()
