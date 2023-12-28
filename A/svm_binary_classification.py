from tqdm import tqdm
import numpy as np

import medmnist
from medmnist import INFO, Evaluator
from utils.base_logger import get_logger
import torchvision.transforms as transforms
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SVMBinaryClassification:

    def __init__(self):
        self.logger = get_logger()
        self.data_class = medmnist.PneumoniaMNIST
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.download_folder = "./Datasets/PneumoniaMNIST"
    


    def download_images(self): 
        self.logger.info('Downloading images PneumoniaMNIST ..... ')
        train_dataset = self.data_class(split='train', transform=self.data_transform, download=True, root = self.download_folder, as_rgb= False)
        test_dataset = self.data_class(split='test', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= False)

    def transform_with_pca(self, image_data): 
        self.logger.info('Reducing features with PCA .....')
        pca = PCA(n_components=600, svd_solver='auto', whiten=True)
        pca.fit(image_data)
        return pca.transform(image_data)
    
    def normalise_data(self, image_data): 
        standard_scalar = StandardScaler()
        return standard_scalar.fit_transform(image_data)
    

    def train_model(self):
        self.download_images()
        self.logger.info('Starting training of model using SVM ..... ')
        with np.load("./Datasets/PneumoniaMNIST/pneumoniamnist.npz") as data:
            train_images = data['train_images']
            train_labels = data['train_labels']

            val_images = data['val_images']
            val_labels = data['val_labels']

            test_images = data['test_images']
            test_labels = data['test_labels']
    
        train_images = train_images.reshape(train_images.shape[0], -1)

        print('Feature matrix shape is: ', train_images.shape)

        train_images = self.normalise_data(train_images)
        compressed_train_images = self.transform_with_pca(train_images)

        print('PCA matrix shape is: ', compressed_train_images.shape)

        
        self.logger.info(compressed_train_images)

    
        train_labels = train_labels.ravel()
        print(train_labels)
        print(np.shape(train_labels))
        svm_model = svm.SVC(kernel='linear', C = 50.0)
        svm_model.fit(train_images, train_labels)

      
        test_images = test_images.reshape(test_images.shape[0], -1)

       
        test_images = self.normalise_data(test_images)
        compressed_test_images = self.transform_with_pca(test_images)

        predict_values = svm_model.predict(test_images)


        print('===== Accuracy Score ===== ')
        print(accuracy_score(test_labels.ravel(), predict_values) * 100 )
     

