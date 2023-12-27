from B.PathMNIST_multi_classification import PathMNISTImageMultiClassification



def main():
    image_multi_class = PathMNISTImageMultiClassification()
    image_multi_class.download_images()


if __name__ == "__main__":
    main()