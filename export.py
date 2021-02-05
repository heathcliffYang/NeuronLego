import onnx
import torch
import numpy as np

from model.model import CNN
from utils.dataset import MaskedFaceDataset, create_dataloader

# Load model
model = CNN()

rootDir = '/home/ginny/Projects/dataset/masked_face_dataset/'
PATH = '/home/ginny/Projects/models/masked_face_classifier/32_20.pt'

model.load_state_dict(torch.load(PATH))
model.eval()

dataset_test = MaskedFaceDataset(rootDir+'test/')

img = torch.tensor(np.array([dataset_test[0]['image']])).clone().detach().permute(0,3,1,2).type(torch.FloatTensor)

print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
f = PATH.replace('.pt', '.onnx')  # filename
torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                  output_names=['output'])

# Checks
onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
# print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
print('ONNX export success, saved as %s' % f)
