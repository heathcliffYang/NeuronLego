import cv2
from torch.utils.data import Dataset, DataLoader
from utils.image import get_image, to_tensor
from criterions.loss import CosineLoss
import numpy as np

import os
import sys
import random

image_suffix = ['jpg', 'JPG', 'JPEG', 'jpeg', 'png', 'PNG', 'bmp']

missing_list = [199 ,
1401 ,
2432 ,
2583 ,
2920 ,
4416 ,
4700 ,
4867 ,
6531 ,
7055 ,
11477 ,
11793 ,
15153 ,
16530 ,
17291 ,
17702 ,
18599 ,
18654 ,
19057 ,
19205 ,
20344 ,
22629 ,
24184 ,
24222 ,
24822 ,
25188 ,
25887 ,
26135 ,
26928 ,
27185 ,
28434 ,
29730 ,
30782 ,
31125 ,
31958 ,
33692 ,
34715 ,
39459 ,
41080 ,
41897 ,
42291 ,
42294 ,
42297 ,
43032 ,
44200 ,
44503 ,
44681 ,
46252 ,
46408 ,
47456 ,
47835 ,
48286 ,
48459 ,
50199 ,
50254 ,
50762 ,
52317 ,
53207 ,
53216 ,
53309 ,
56809 ,
61323 ,
63204 ,
63310 ,
63311 ,
63357 ,
63507 ,
63807 ,
64842 ,
67062 ,
68920 ,
69706 ,
69956 ,
72007 ,
72776 ,
73096 ,
74091 ,
74881 ,
75398 ,
76230 ,
76881 ,
77942 ,
79609 ,
80480 ,
82592 ,
82909 ,
83776 ,
85659 ,
85715 ,
87596 ,
88404 ,
88537 ,
88951 ,
89515 ,
89764 ,
89946 ,
90515 ,
92651 ,
93057 ,
94491 ,
96502 ,
97029 ,
97810 ,
101787 ,
102061 ,
108320 ,
110546 ,
110774 ,
113617 ,
113985 ,
115470 ,
118004 ,
118933 ,
119572 ,
120599 ,
121050 ,
122805 ,
123466 ,
123468 ,
123505 ,
124247 ,
125520 ,
126707 ,
127033 ,
131065 ,
132615 ,
132740 ,
133222 ,
136832 ,
137880 ,
139286 ,
140439 ,
143009 ,
143376 ,
143541 ,
145265 ,
145340 ,
147930 ,
148072 ,
149120 ,
149846 ,
150051 ,
150895 ,
152534 ,
153323 ,
153819 ,
154156 ,
154288 ,
155280 ,
157514 ,
157799 ,
159400 ,
159886 ,
160341 ,
163146 ,
163906 ,
164273 ,
166618 ,
169941 ,
170965 ,
171615 ,
173464 ,
174224 ,
174759 ,
174980 ,
175197 ,
176165 ,
176407 ,
177221 ,
177913 ,
178807 ,
179577 ,
180108 ,
180786 ,
180858 ,
181166 ,
181885 ,
182123 ,
182979 ,
183734 ,
183917 ,
191114 ,
191321 ,
192086 ,
195995 ,
196426 ,
198447 ,
198603 ,
200472]

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
        self.img_id = dict()
        self.mini_batch = mini_batch
        self.net = net
        with open(self.folder+"/Anno/identity_CelebA.txt", 'r') as f:
            for i in f:
                idx = i.split(" ")[0]
                id = int(i.split(" ")[1])
                idx_num = int(idx.split(".")[0])
                if idx_num in missing_list:
                    continue
                self.img_id[idx] = id
        self.img_id_list = list(self.img_id)
        self.criterion = CosineLoss()

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):              # 6 digit + ".jpg" id
        input_image_id = self.img_id_list[idx]
        img_path = self.folder + '/Img/train/' + input_image_id
        image = get_image(img_path)
        v = self.net(to_tensor(image))
        # print("sample:", input_image_id)
        # positive
        positive_sample_list = []
        for i in self.img_id_list:
            if self.img_id[i] == self.img_id[input_image_id]: ## the same person
                positive_sample_list.append(i)
        positive_sample = random.choice(positive_sample_list)
        img_path = self.folder + '/Img/train/' + positive_sample
        positive = get_image(img_path)
        # print("positive:", positive_sample)
        # negative
        negative_sample_list = random.choices(self.img_id_list, k=self.mini_batch)
        max_negative = None
        max_dis = 0
        for i in negative_sample_list:
            if self.img_id[i] == self.img_id[input_image_id]: ## need to filter out the same person
                continue
            img_path = self.folder + '/Img/train/' + i
            # print("candidate negative:", i)
            negative = get_image(img_path)
            v_n = self.net(to_tensor(negative))
            dis = self.criterion.forward(v, v_n, -1)
            if dis >= max_dis:
                max_dis = dis
                max_negative = negative
        # FIX: if all sample negatives have the same id with sample...
        # triplet sample
        sample = {'image': image, 'positive': positive, 'negative': negative}
        return sample


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size, shuffle=True)
