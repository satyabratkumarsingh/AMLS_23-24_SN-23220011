from tqdm import tqdm
import numpy as np

import medmnist
from medmnist import INFO, Evaluator
from utils.base_logger import get_logger
import torchvision.transforms as transforms


class PathMNISTImageMultiClassification:
    def __init__(self):
        self.logger = get_logger()
        self.data_class = medmnist.PathMNIST
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.download_folder = "./Datasets/PathMNIST"

    def download_images(self):
        self.logger.info('Downloading images ..... ')
        train_dataset = self.data_class(split='train', transform=self.data_transform, download=True, root = self.download_folder, as_rgb= True)
        test_dataset = self.data_class(split='test', transform=self.data_transform, download=True,  root = self.download_folder, as_rgb= True)
        train_metrix, train_label = train_dataset[0]
        self.logger.info(train_metrix)
        self.logger.info(train_metrix.shape)

    





