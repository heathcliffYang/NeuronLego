import cv2
import numpy as np
import torch

def get_image(img_path):
    image = cv2.imread(img_path)
    empty_image = np.zeros((75, 149, 3))
    # crop upper part
    image = image[:int(image.shape[0]/2),:, :]
    ratio = min(149/image.shape[1], 75/image.shape[0])
    h = int(image.shape[0]*ratio)
    w = int(image.shape[1]*ratio)
    image = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
    pad_w = int((149 - w) / 2)
    pad_h = int((75 - h) / 2)
    empty_image[pad_h:pad_h+h, pad_w:pad_w+w, :] = image
    empty_image = empty_image / 255.
    return empty_image

def to_tensor(image):
    return torch.from_numpy(np.array([image])).clone().detach().permute(0,3,1,2).type(torch.cuda.FloatTensor)
