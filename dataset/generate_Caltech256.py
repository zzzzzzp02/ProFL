import os
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import albumentations
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Caltech256/"


# custom dataset
class ImageDataset(Dataset):
    def __init__(self, images, labels=None):
        self.X = images
        self.y = labels

        # apply augmentations
        self.aug = albumentations.Compose([
            albumentations.Resize(128, 128, always_apply=True),
            albumentations.CenterCrop(112, 112, always_apply=True)
        ])

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = image.convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'target': torch.tensor(label, dtype=torch.long)
        }


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # load raw data
    root_dir = dir_path + 'rawdata/256_ObjectCategories'
    # get all the folder paths
    all_paths = os.listdir(root_dir)

    # create a DataFrame
    rawdata = pd.DataFrame()

    images = []
    labels = []
    counter = 0
    for folder_path in tqdm(all_paths, total=len(all_paths)):
        # get all the image names in the particular folder
        image_paths = os.listdir(f"{root_dir}/{folder_path}")
        # get the folder as label
        label = folder_path.split('.')[-1]

        if label == 'clutter':
            continue

        # save image paths in the DataFrame
        for image_path in image_paths:
            if image_path.split('.')[-1] == 'jpg':
                rawdata.loc[counter, 'image_path'] = f"{root_dir}/{folder_path}/{image_path}"
                labels.append(label)
                counter += 1

    labels = np.array(labels)
    # one-hot encode the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # add the image labels to the dataframe
    for i in range(len(labels)):
        index = np.argmax(labels[i])
        rawdata.loc[i, 'target'] = int(index)

    # shuffle the dataset
    rawdata = rawdata.sample(frac=1).reset_index(drop=True)

    # load Dataset
    X = rawdata.image_path.values  # image paths
    y = rawdata.target.values  # targets

    all_dataset = ImageDataset(X, y)
    all_dataloader = DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False)

    # Get Caltech-256 data
    dataset_image = []
    dataset_label = []
    for dataall in all_dataloader:
        dataset_image.extend(dataall['image'].cpu().detach().numpy())
        dataset_label.extend(dataall['target'].cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=3)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
