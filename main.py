from B.PathMNIST_multi_classification import PathMNISTImageMultiClassification
from A.svm_binary_classification import SVMBinaryClassification


def main():
    svm_binary_classification = SVMBinaryClassification()
    svm_binary_classification.train_model()

    #image_multi_class = PathMNISTImageMultiClassification()
    #image_multi_class.download_images()


if __name__ == "__main__":
    main()