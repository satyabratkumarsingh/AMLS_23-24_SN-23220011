from B.PathMNIST_multi_classification import PathMNISTImageMultiClassification
from A.BinaryClassification import BinaryClassification
import numpy as np

def main():
    binary_classification = BinaryClassification()

    binary_classification.train_models()

    #image_multi_class = PathMNISTImageMultiClassification()
    #image_multi_class.download_images()


if __name__ == "__main__":
    main()