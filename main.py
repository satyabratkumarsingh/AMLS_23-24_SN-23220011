
from  A.binary_classification import BinaryClassification
from B.multi_classification import MultiClassification
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from utils.base_logger import get_logger
import time


def main():
    start = time.process_time()
    binary_classification = BinaryClassification()

    #Trains and tests binary classification model
    binary_classification.train_models()

    multi_class = MultiClassification()
    
    #Trains and tests multi classification model
    multi_class.train_model()
    multi_class.test_model()
    end_time = time.process_time() - start
    get_logger().info(f'************* Time taken to run both Binary and Multi Classification Model is ******** ', end_time)


if __name__ == "__main__":
    main()