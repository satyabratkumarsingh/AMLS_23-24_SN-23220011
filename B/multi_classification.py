from tqdm import tqdm
import numpy as np
import torch
import medmnist
from medmnist import INFO, Evaluator
from utils.base_logger import get_logger
import torchvision.transforms as transforms
from B.convolutional_neural_network import ConvolutionalNeuralNetwork
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, recall_score, auc
import time

class MultiClassification:
    def __init__(self):
        self.logger = get_logger()
        self.data_class = medmnist.PathMNIST
        medmnist.PathMNIST
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.download_folder = "./Datasets/PathMNIST"
        # define training hyperparameters
        self.LEARING_RATE = 0.08
        self.BATCH_SIZE = 64
        self.EPOCHS = 50
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.classification_size = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"



    def download_images(self):
        self.logger.info('====== Downloading images ..... ')
        self.train_dataset = self.data_class(split='train', transform=self.data_transform, download=True, root = self.download_folder, as_rgb= True)
        self.test_dataset = self.data_class(split='test', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= True)
        self.validation_dataset = self.data_class(split='val', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= True)
        self.classification_size = np.unique(self.train_dataset.labels.ravel()).size
        self.logger.info(f' ==== There are %s different classification in the training set. ==== ',  self.classification_size)

        self.logger.info(f' ==== Training Data Size %s ==== ',  len(self.train_dataset))
        self.logger.info(f' ==== Validation Data Size %s. ==== ',  len(self.validation_dataset))

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_loader =  DataLoader(dataset=self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_loader =  DataLoader(dataset=self.validation_dataset, batch_size=self.BATCH_SIZE, shuffle=True)


    def train_model(self):
        start = time.process_time()
        self.download_images()
        cnn_model = ConvolutionalNeuralNetwork(self.classification_size)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn_model.parameters(), lr=self.LEARING_RATE)

        for epoch in range(self.EPOCHS):
            for inputs, targets in tqdm(self.val_loader):
            # forward + backward + optimize
                optimizer.zero_grad()
                outputs = cnn_model(inputs)
                targets = targets.squeeze().long()
                loss = loss_function(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad(): #Don't track gradients
                cnn_model.eval()
                correct = 0
                total = 0
                all_val_loss = []
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = cnn_model(images)
                    total += labels.size(0)
                    # calculate predictions
                    predicted = torch.argmax(outputs, dim=1)
                    # calculate actual values
                    correct += (predicted == labels).sum().item()
                    # calculate the loss
                    labels = labels.squeeze().long()

                    all_val_loss.append(loss_function(outputs, labels).item())
                # calculate val-loss
                mean_val_loss = sum(all_val_loss) / len(all_val_loss)
                # calculate val-accuracy
                mean_val_acc = 100 * (correct / total)
                print(
                    'Epoch [{}/{}], Loss: {:.4f}, Val-loss: {:.4f}, Val-acc: {:.1f}%'.format(
                        epoch+1, self.EPOCHS, loss.item(), mean_val_loss, mean_val_acc
                    )
                )
        torch.save(cnn_model, 'cnn.pt')
        self.logger.info(f' ============== The time taken to train CNN is %s ==========',  time.process_time() - start)

                

    def test_model(self):
        """ Load saved model and evaluate the performance """
        start = time.process_time()
        cnn_model = torch.load('cnn.pt')
        cnn_model.eval()
        all_labels = torch.empty(0, dtype=torch.int)
        all_predicted_labels = torch.empty(0, dtype=torch.int)
        
        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = cnn_model(images)
            predicted_labels = torch.argmax(outputs, dim=1)
            
            all_predicted_labels = torch.cat([all_predicted_labels, predicted_labels], dim=0)
            all_labels = torch.cat([all_labels, torch.flatten(labels)], dim=0)

    
        self.logger.info(f' ============== Actual Labels ==========')
        self.logger.info(all_labels.numpy())
        self.logger.info(f' ============== Predicted Labels ==========')
        self.logger.info(all_predicted_labels.numpy())
        class_report = classification_report(all_labels.numpy(), all_predicted_labels.numpy(), output_dict=True)
        self.logger.info(f' ============== Classification Report ==========')
        self.logger.info(class_report)
        self.logger.info(f' ============== The time taken to TEST CNN is %s ==========',  time.process_time() - start)

        