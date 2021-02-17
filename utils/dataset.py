import cv2
from torch.utils.data import Dataset, DataLoader
from image import get_image
import numpy as np

import os
import sys
import random

image_suffix = ['jpg', 'JPG', 'JPEG', 'jpeg', 'png', 'PNG', 'bmp']

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
        image = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2RGB)
        # normalize
        image = image / 255.
        sample = {'image': image, 'label': self.label_list[idx], 'filepath': self.image_file_list[idx]}
        return sample


class UpperFaceDataset(Dataset):
    """
    For CelebA dataset
    """
    def __init__(self, rootDir, mini_batch, net):
        self.folder = rootDir
        self.img_id = []
        self.mini_batch = mini_batch
        self.net = net
        with open(self.folder+"/Anno/identity_CelebA.txt", 'r') as f:
            for i in f:
                id = int(i.split(" ")[1])
                self.img_id.append(id)
                print(id)
                break

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):              # 6 digit + ".jpg" id
        img_path = self.folder + '/Img/img_align_celeba/' + format(str(idx), "0>6") + ".jpg"
        image = get_image(img_path)
        v = self.net(v)
        # positive
        positive_sample_list = []
        for i in range(len(self.img_id)):
            if self.img_id[i] == self.img_id[idx]:
                positive_sample_list.append(i)
        positive_sample = random.choice(positive_sample_list)
        img_path = self.folder + '/Img/img_align_celeba/' + format(str(positive_sample), "0>6") + ".jpg"
        positive = get_image(img_path)
        # negative
        negative_sample_list = np.random.uniform(0, len(self.img_id)-1, self.mini_batch)
        min_negative = None
        min_dis = sys.maxint
        for i in negative_sample_list:
            if self.img_id[i] == self.img_id[idx]:
                continue
            img_path = self.folder + '/Img/img_align_celeba/' + format(str(i), "0>6") + ".jpg"
            negative = get_image(img_path)
            v_n = self.net(negative)
            if v_n < min_dis:
                min_dis = v_n
                min_negative = negative
        # FIX: if all sample negatives have the same id with sample...
        # triplet sample
        sample = {'image': image, 'label': self.img_id[idx], 'positive': positive, 'negative': negative}
        return sample


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size, shuffle=True)
