import onnx
import torch
import numpy as np

from models.model import CNN
from utils.dataset import MaskedFaceDataset, create_dataloader

# Load model
model = CNN()

rootDir = '/home/ginny/Projects/dataset/masked_face_dataset/'
PATH = '/home/ginny/Projects/models/masked_face_classifier/32_11.pt'

model.load_state_dict(torch.load(PATH))
model.eval()

dataset_test = MaskedFaceDataset(rootDir+'test/')
dataloader_test = create_dataloader(dataset_test, batch_size=64)

acc  = 0
for i_batch, sample_batched in enumerate(dataloader_test):
    input = sample_batched['image'].clone().detach().permute(0,3,1,2).type(torch.FloatTensor)
    prediction, mu, logvar = model(input)
    gt = sample_batched['label'].view(-1,1).type(torch.FloatTensor)
    file_path_list = sample_batched['filepath']

    positive = 0
    negative = 0
    for p, i, g, fp in zip(prediction, torch.sigmoid(prediction), gt, file_path_list):
        print("%.2f, %.2f, %.2f, %s"%(p.item(), i.item(), g.item(), fp))
        if i > 0.5 and g > 0.5:
            acc += 1
        if i < 0.5 and g < 0.5:
            acc += 1
print("Test - accuracy: %.2f"%(acc / len(dataset_test)))

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
