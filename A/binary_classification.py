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
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import time

""" This class combines all the functionlaity for binary classification of the dataset """
class BinaryClassification:

    def __init__(self):
        """ The constructor, initializing all models to try """
        self.logger = get_logger()
        self.data_class = medmnist.PneumoniaMNIST
        self.data_transform = transforms.Compose([
            transforms.ToTensor()
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
                         'n_estimators': list(range(1,5))
                },
                'confusion_matrix_file' : './plots/A/confusion_matrix_random_forest.jpeg'
            }
        }

    def plot_confusion_matrix(self, y_true, y_pred, filename, labels):
       
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    if (i == 0 ):
                        label  = 'TN'
                    else:
                       label  = 'TP' 
                    annot[i, j] = '%.2f%%\n%d/%d\n %s' % (p, c, s, label)
              
                else:
                    if (j == 0 ):
                        label  = 'FN'
                    else:
                       label  = 'FP' 
                    annot[i, j] = '%.2f%%\n%d\n%s' % (p, c, label)
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm = cm * 100
        sns.set(font_scale=1.4)
        plot =  sns.heatmap(cm, annot=annot, fmt='',cbar=True, cbar_kws={'format':PercentFormatter()}, cmap="Blues")
        figure = plot.get_figure()    
        figure.savefig(filename)
        figure.clf()

    def train_models(self): 
        """ First download and then train models, the models and the options are defined in  self.models 
        """
        start = time.process_time()
        self.download_images()
        self.logger.info('Starting training of model using SVM, Random Forest etc..... ')
        with np.load("./Datasets/PneumoniaMNIST/pneumoniamnist.npz") as data:
            train_images = data['train_images']
            train_labels = data['train_labels']

            val_images = data['val_images']
            val_labels = data['val_labels']

            test_images = data['test_images']
            test_labels = data['test_labels']
        
        self.logger.info(f' ******* The shape of training dataset is : %s ****', train_images.shape)
        train_images = train_images.reshape(train_images.shape[0], -1)
        self.logger.info(f' ========== The shape of training dataset after transformation : %s ****', train_images.shape)
        train_images = self.normalise_data(train_images)

        # We are commenting it, as using PCA for SVM brough down accuracy score from 97 % to  91 %
        # It also impacted negarively the confusion matrix.
        # self.logger.info('==============Trying to reduce dimensions ================')
        # compressed_train_images = self.transform_with_pca(train_images)
        # print('PCA matrix shape is: ', compressed_train_images.shape)

        train_labels = train_labels.ravel()
        best_model_n_scores = []
        for model_name, model in self.models.items(): 
            grid_search_cv = GridSearchCV(model['model'], model['params'], cv = 5, scoring='accuracy')
            
            self.logger.info(f'Fitting model : %s', model_name)
            
            grid_search_cv.fit(train_images, train_labels)

            
            self.logger.info(f'Model : %s, fitted, best score is %s', model_name, grid_search_cv.best_score_)

            val_images = val_images.reshape(val_images.shape[0], -1)
            val_images = self.normalise_data(val_images)
            grid_predictions = grid_search_cv.best_estimator_.predict(val_images)
            
            class_report = classification_report(val_labels.ravel(), grid_predictions, output_dict=True)
            self.logger.info('========= Classification Report for model {%s}==========', model['model'])
            self.logger.info(class_report)
            
            macro_precision =  class_report['macro avg']['precision'] 
            macro_recall = class_report['macro avg']['recall']    
            macro_f1 = class_report['macro avg']['f1-score']
            macro_accuracy = class_report['accuracy']

            self.logger.info('========= AUC ==========')
            auc_ = auc(val_labels.ravel(), grid_predictions)
            self.logger.info(auc_)

            self.logger.info('========= Confusion Matrix ==========')
            conf_matrix = confusion_matrix(val_labels.ravel(), grid_predictions)
            self.logger.info(conf_matrix)
            self.plot_confusion_matrix(val_labels.ravel(), grid_predictions, model['confusion_matrix_file'], labels = grid_search_cv.classes_)
            #self.plot_confusion_matrix(conf_matrix, model['confusion_matrix_file'])

            best_model_n_scores.append({
                 'model_name' : model_name, 
                 'best_model' : grid_search_cv.best_estimator_,
                 'best_score_f1': macro_f1,
                 'best_score_precision': macro_precision,
                 'best_score_recall': macro_recall,
                 'best_score_accuracy': macro_accuracy,
                 'best_params': grid_search_cv.best_params_,
             })
            
        self.logger.info('========= The best models and scores are  ==========')    
        self.logger.info(best_model_n_scores)
        self.logger.info('========= Now trying to find best model  ==========')    
        best_score_fields = ['best_score_f1', 'best_score_accuracy']
        best_scored_model = max(best_model_n_scores, key=lambda x: tuple(x[field] for field in best_score_fields))['best_model']

        self.logger.info('Now we know the best model and parameters, retraining with those .......')
        # We found best model as 'svm', 'parms': SVC(C=11, gamma='auto', kernel = 'rbf)
        # We need to train it again with entire training set data 
        svm_model = svm.SVC(gamma='auto',  kernel='rbf', C= 11)
        svm_model.fit(train_images, train_labels)
        test_images = test_images.reshape(test_images.shape[0], -1)
        test_images = self.normalise_data(test_images)
        test_predictions = svm_model.predict(test_images)
        conf_matrix = confusion_matrix(test_labels.ravel(), test_predictions)
        test_class_report = classification_report(test_labels.ravel(), test_predictions, output_dict=True)
        self.logger.info('************ TEST Classification Report ************')
        self.logger.info(test_class_report)
        self.plot_confusion_matrix(test_labels.ravel(), y_pred = test_predictions, filename='./plots/A/confusion_matrix_test_data_set.jpeg', labels = svm_model.classes_ )
    
        self.logger.info(f' ============== The time taken to run Binary classification model is %s ==========',  time.process_time() - start)
        
        
    def download_images(self): 
        """ Download images and save into /Datasets/PneumoniaMNIST """
        self.logger.info('Downloading images PneumoniaMNIST ..... ')
        train_dataset = self.data_class(split='train', transform=self.data_transform, download=True, root = self.download_folder, as_rgb= False)
        test_dataset = self.data_class(split='test', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= False)

    def transform_with_pca(self, image_data): 
        """ Apply PCA to reduce dimensions, we reduced the dimensions from  """
        self.logger.info('Reducing features with PCA .....')
        pca = PCA(n_components=600, svd_solver='auto', whiten=True)
        pca.fit(image_data)
        return pca.transform(image_data)
    
    def normalise_data(self, image_data): 
        """ To normalize data, the formula used in StandardScalar is z = (x - u) / s  """
        standard_scalar = StandardScaler()
        return standard_scalar.fit_transform(image_data)
    
    
     

