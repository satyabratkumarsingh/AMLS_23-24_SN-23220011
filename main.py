from B.PathMNIST_multi_classification import PathMNISTImageMultiClassification
from  A.binary_classification import BinaryClassification
from B.multi_classification import MultiClassification
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def main():
    binary_classification = BinaryClassification()

    binary_classification.train_models()

    multi_class = MultiClassification()
    #image_multi_class = PathMNISTImageMultiClassification()
    #multi_class.train_model()
    

if __name__ == "__main__":
    main()