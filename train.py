from utils.dataset import MaskedFaceDataset, create_dataloader
from criterions.loss import Loss
from models.model import CNN
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch

"""
training hyperparameter
"""
epoch = 1000
lr = 0.001
momentum = 0.9
lf = lambda epoch: 0.95 ** epoch

"""
Load input data
"""
rootDir = '/home/ginny/Projects/dataset/masked_face_dataset/'
dataset_train = MaskedFaceDataset(rootDir+'train/')
dataloader_train = create_dataloader(dataset_train, batch_size=64)
dataset_test = MaskedFaceDataset(rootDir+'test/')
dataloader_test = create_dataloader(dataset_test, batch_size=64)

"""
build model
"""
cnn = CNN()

"""
Loss and optimizer
"""
criterion = Loss()
optimizer = optim.SGD(cnn.parameters(), lr, momentum)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lf])

"""
training
"""
for e in range(epoch):
    cnn.train()
    """
    train
    """
    loss_total = 0
    acc  = 0
    for i_batch, sample_batched in enumerate(dataloader_train):
        # print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())

        optimizer.zero_grad()

        input = sample_batched['image'].clone().detach().permute(0,3,1,2).type(torch.FloatTensor)
        # print(input.type())

        prediction, mu, logvar = cnn(input)
        gt = sample_batched['label'].view(-1,1).type(torch.FloatTensor)

        # print(prediction.type(), gt.type(), prediction.shape)
        loss = criterion.forward(prediction, gt)
        loss.backward()
        optimizer.step()


        loss_total += loss.item()
        positive = 0
        negative = 0
        for i, g in zip(torch.sigmoid(prediction), gt):
            if i > 0.5 and g > 0.5:
                acc += 1
            if i < 0.5 and g < 0.5:
                acc += 1
    print("Epoch %d - loss: %.3f - hits: %.2f"%(e, loss_total / len(dataset_train), acc / len(dataset_train)))
    cnn.eval()
    """
    test
    """
    loss_total = 0
    acc  = 0
    for i_batch, sample_batched in enumerate(dataloader_test):
        optimizer.zero_grad()
        input = sample_batched['image'].clone().detach().permute(0,3,1,2).type(torch.FloatTensor)

        prediction, mu, logvar = cnn(input)
        gt = sample_batched['label'].view(-1,1).type(torch.FloatTensor)

        loss = criterion.forward(prediction, gt)
        loss.backward()
        optimizer.step()


        loss_total += loss.item()
        positive = 0
        negative = 0
        for i, g in zip(torch.sigmoid(prediction), gt):
            if i > 0.5 and g > 0.5:
                acc += 1
            if i < 0.5 and g < 0.5:
                acc += 1
    print("Test - loss: %.3f - hits: %.2f"%(loss_total / len(dataset_test), acc / len(dataset_test)))

    PATH = '/home/ginny/Projects/models/masked_face_classifier/64_%d.pt'%(e)

    torch.save(cnn.state_dict(), PATH)
