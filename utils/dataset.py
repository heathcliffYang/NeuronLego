import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os

image_suffix = ['jpg', 'JPG', 'JPEG', 'jpeg', 'png', 'PNG']

class MaskedFaceDataset(Dataset):

    def __init__(self, rootDir):
        self.folder_label_list = [0, 1]
        self.folder_list = ["masked", "whole"]
        self.image_file_list = []
        self.label_list = []
        for folder, label in zip(self.folder_list, self.folder_label_list):
            for root, dirs, files in os.walk(rootDir+"/"+folder):
                for f in files:
                    for suffix in image_suffix:
                        if suffix in f:
                            self.image_file_list.append(root+"/"+f)
                            self.label_list.append(label)
                            break
        print("Dataset init:\n  - images: %d, labels: %d\n"%(len(self.image_file_list), len(self.label_list)))

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_file_list[idx])
        # print(self.image_file_list[idx])
        image = cv2.resize(image, (64, 64))
        # normalize
        image_mean = np.mean(image)
        image_std = np.std(image)
        image = (image - image_mean)/image_std
        # print(np.mean(image), np.std(image))
        sample = {'image': image, 'label': self.label_list[idx]}
        return sample


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size, shuffle=True)
