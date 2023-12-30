from tqdm import tqdm
import numpy as np

import medmnist
from medmnist import INFO, Evaluator
from utils.base_logger import get_logger
import torchvision.transforms as transforms
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score, auc
import seaborn as sns

class BinaryClassification:

    def __init__(self):
        self.logger = get_logger()
        self.data_class = medmnist.PneumoniaMNIST
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.download_folder = "./Datasets/PneumoniaMNIST"
        self.models = {
            'svm' : {
                'model' : svm.SVC(gamma='auto'),
                'params': {
                        'C' : list(range(10,30)),
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                },
                'confusion_matrix_file' : './plots/A/confusion_matrix_svm.jpeg'
            }, 
            'random_forest': {
                'model' : RandomForestClassifier(),
                'params': {
                         'n_estimators': [1,5]
                },
                'confusion_matrix_file' : './plots/A/confusion_matrix_random_forest.jpeg'
            }
        }
    
    def plot_confusion_matrix(self, conf_matrix, file_name):
        group_names = ['True Negative','False Positive','False Negative','True Positive']
        group_counts = ['{0:0.0f}'.format(value) for value in
                        conf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        plot = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
        figure = plot.get_figure()    
        figure.savefig(file_name)


    def train_models(self): 
        self.download_images()
        self.logger.info('Starting training of model using SVM, Random Forest etc..... ')
        with np.load("./Datasets/PneumoniaMNIST/pneumoniamnist.npz") as data:
            train_images = data['train_images']
            train_labels = data['train_labels']

            val_images = data['val_images']
            val_labels = data['val_labels']

            test_images = data['test_images']
            test_labels = data['test_labels']

        train_images = train_images.reshape(train_images.shape[0], -1)
        train_images = self.normalise_data(train_images)
        train_labels = train_labels.ravel()
        scores = []
        for model_name, model in self.models.items(): 
            grid_search_cv = GridSearchCV(model['model'], model['params'], cv = 5, return_train_score= False)
            self.logger.info(f'Fitting model : %s', model_name)
            grid_search_cv.fit(train_images, train_labels)
            scores.append({
                 'model' : model_name, 
                 'best_score': grid_search_cv.best_score_,
                 'best_params': grid_search_cv.best_params_,
             })
            self.logger.info(f'Model : %s, fitted, best score is %s', model_name, grid_search_cv.best_score_)

            val_images = val_images.reshape(val_images.shape[0], -1)
            val_images = self.normalise_data(val_images)
            grid_predictions = grid_search_cv.best_estimator_.predict(val_images)
            
            class_report = classification_report(val_labels.ravel(), grid_predictions)
            self.logger.info('========= Classification Report ==========')
            self.logger.info(class_report)

            self.logger.info('========= AUC ==========')
            auc_ = auc(val_labels.ravel(), grid_predictions)
            self.logger.info(auc_)

            self.logger.info('========= Confusion Matrix ==========')
            conf_matrix = confusion_matrix(val_labels.ravel(), grid_predictions)
            self.logger.info(conf_matrix)
            self.plot_confusion_matrix(conf_matrix, model['confusion_matrix_file'])
            
            
            
        self.logger.info(scores)

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
    
    def train_svm_with_cross_validation(self):
        self.download_images()
        self.logger.info('Starting training of model using SVM Cross validation..... ')
        with np.load("./Datasets/PneumoniaMNIST/pneumoniamnist.npz") as data:
            train_images = data['train_images']
            train_labels = data['train_labels']

            val_images = data['val_images']
            val_labels = data['val_labels']

            test_images = data['test_images']
            test_labels = data['test_labels']

        train_images = train_images.reshape(train_images.shape[0], -1)
        #train_images = self.normalise_data(train_images)
        
        print(train_labels)
        svm_model = svm.SVC(gamma='auto')
        c_values = list(range(1,30))
        kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
        grid_search_cv = RandomizedSearchCV(svm_model, {'C': c_values, 'kernel': kernel_values}, cv=5,
                                      return_train_score=False)
        grid_search_cv.fit(train_images, train_labels.ravel())
        df_scores = pd.DataFrame(grid_search_cv.cv_results_)
        self.logger.info(df_scores[['mean_test_score', 'param_C', 'param_kernel']])
    

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
        
        cross_val_score()
      
        test_images = test_images.reshape(test_images.shape[0], -1)

       
        test_images = self.normalise_data(test_images)
        compressed_test_images = self.transform_with_pca(test_images)

        predict_values = svm_model.predict(test_images)


        print('===== Accuracy Score ===== ')
        print(accuracy_score(test_labels.ravel(), predict_values) * 100 )
     

