from utils.dataset import UpperFaceDataset, create_dataloader
from criterions.loss import CosineLoss
from models.inceptionresnetv2_half import InceptionResNetV2
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
mini_batch = 32

"""
build model
"""
model = InceptionResNetV2().cuda()

"""
Load input data
"""
rootDir = '/home/ginny/Projects/dataset/faces/'
dataset_train = UpperFaceDataset(rootDir, 32, model)
dataloader_train = create_dataloader(dataset_train, batch_size=12)
# dataset_test = UpperFaceDataset(rootDir, 32, cnn)
# dataloader_test = create_dataloader(dataset_test, batch_size=64)

"""
Loss and optimizer
"""
criterion = CosineLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lf])
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

"""
training
"""
print("start training")
for e in range(epoch):
    model.train()
    """
    train
    """
    loss_total = 0
    acc  = 0
    for i_batch, sample_batched in enumerate(dataloader_train):
        print("batch:", i_batch)
        optimizer.zero_grad()
        input = sample_batched['image'].clone().detach().permute(0,3,1,2).type(torch.cuda.FloatTensor)
        negative = sample_batched['negative'].clone().detach().permute(0,3,1,2).type(torch.cuda.FloatTensor)
        positive = sample_batched['positive'].clone().detach().permute(0,3,1,2).type(torch.cuda.FloatTensor)
        embedding = model(input)
        embedding_n = model(negative)
        embedding_p = model(positive)
    #     gt = sample_batched['label'].view(-1,1).type(torch.FloatTensor)
        loss = criterion.forward(embedding, embedding_n, -1)
        loss += criterion.forward(embedding, embedding_p, 1)

        loss.backward()
        optimizer.step()
        print("loss : %.3f"%(loss.item()))
        print("negative:", cos(embedding, embedding_n))
        print("positive:", cos(embedding, embedding_p))

        loss_total += loss.item()
    print("Total loss : %.3f"%(loss_total))
    #     positive = 0
    #     negative = 0
    #     for i, g in zip(torch.sigmoid(prediction), gt):
    #         if i > 0.5 and g > 0.5:
    #             acc += 1
    #         if i < 0.5 and g < 0.5:
    #             acc += 1
    # print("Epoch %d - loss: %.3f - hits: %.2f"%(e, loss_total / len(dataset_train), acc / len(dataset_train)))
    # cnn.eval()
    # """
    # test
    # """
    # loss_total = 0
    # acc  = 0
    # for i_batch, sample_batched in enumerate(dataloader_test):
    #     optimizer.zero_grad()
    #     input = sample_batched['image'].clone().detach().permute(0,3,1,2).type(torch.FloatTensor)
    #
    #     prediction, mu, logvar = cnn(input)
    #     gt = sample_batched['label'].view(-1,1).type(torch.FloatTensor)
    #     loss = criterion.forward(prediction, gt)
    #
    #     loss_total += loss.item()
    #     positive = 0
    #     negative = 0
    #     for i, g in zip(torch.sigmoid(prediction), gt):
    #         if i > 0.5 and g > 0.5:
    #             acc += 1
    #         if i < 0.5 and g < 0.5:
    #             acc += 1
    # print("Test - loss: %.3f - hits: %.2f"%(loss_total / len(dataset_test), acc / len(dataset_test)))
    #
    # PATH = '/home/ginny/Projects/models/masked_face_classifier/32_%d.pt'%(e)
    #
    # torch.save(cnn.state_dict(), PATH)
