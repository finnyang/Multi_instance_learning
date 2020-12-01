import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler


positive_number = 0


class MnistMilNet(nn.Module):
    def __init__(self):
        super(MnistMilNet, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)

        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.feature_extractor = nn.Linear(128*3*3, 128)
        self.weight_generator = nn.Linear(128, 1, bias=False)
        self.classifier = nn.Linear(128, 2, bias=False)
        self.loss_fun = nn.NLLLoss()

    def forward(self, x, target=None, is_train=True):
        instance_number = 10 if is_train else target.shape[0]

        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)

        feature = F.relu(self.feature_extractor(x.reshape(-1, 128*3*3)))
        
        weight = self.weight_generator(feature)
        weight = weight.reshape(-1, instance_number)
        weight = F.softmax(weight, 1)
        weight = weight.reshape(-1, 1)

        feature = feature*weight
        feature = feature.reshape(-1, instance_number, 128).sum(1)
        
        output = F.log_softmax(self.classifier(feature), dim=1)
        
        if is_train:
            target = target.reshape(-1, instance_number)
            target = (((target==positive_number)*1).sum(1)!=0)*1
            loss = self.loss_fun(output, target)
            return loss
        else:
            return output, weight      


def train(model, epoch, optimizer, train_loader):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        loss = model(data.cuda(), target.cuda())
        print("epoch: {}, loss: {}".format(epoch, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    right = 0
    length = 0
    save_index = 0
    max_save = 20

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "test_result")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "test_result"))

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()

        pred, weight = model(data, target, False)
        target = target.reshape(-1, data.shape[0])
        target = (((target==positive_number)*1).sum(1)!= 0)*1
        _, max_index = pred.max(1)
        right += ((max_index == target)*1).sum()
        length += len(pred)
        
        if save_index < max_save:
            visible_data = (data*0.3081+0.1307)*255
            visible_data = visible_data.cpu().numpy()

            temp = []
            for i in range(visible_data.shape[0]):
                temp.append(visible_data[i, 0, :, :])
            image = np.concatenate(temp, 1)
            image = image.astype(np.uint8)
            mask = np.zeros(shape=[image.shape[0]+20, image.shape[1]], dtype=np.uint8)
            mask[0:28, :] = image
            _, index = weight.max(0)
            if max_index == 1:
                mask[30:46, index*28+2:index*28+26] = 255
            im = Image.fromarray(mask)
            im.save(os.path.join(os.path.join(os.path.dirname(__file__), "test_result"), "out{}.jpg".format(save_index)))
        save_index += 1
    print("acc={:.4f}".format(right/length))


def training(max_epoch):
    trainset = datasets.MNIST('./data', 
                              download=True, 
                              train=True, 
                              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                              )
    train_loader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=4)

    model = MnistMilNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001, amsgrad=False)
    sheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "models")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "models"))
    
    for epoch in range(max_epoch):
        sheduler.step(epoch+1)
        train(model, epoch+1, optimizer, train_loader)
        if (epoch+1)%10 == 0:
            save_path = os.path.join(os.path.join(os.path.dirname(__file__), "models"), "{}.pth".format(epoch + 1))
            torch.save(model.state_dict(), save_path)


def testing(epoch):
    testset = datasets.MNIST('./data', 
                             download=True, 
                             train=False, 
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                             )
    test_loader = DataLoader(testset, batch_size=10, shuffle=True, num_workers=4)

    state_dict = torch.load(os.path.join(os.path.join(os.path.dirname(__file__), "models"), "{}.pth".format(epoch)))
    model = MnistMilNet().cuda()
    model.load_state_dict(state_dict)
    test(model, test_loader)


if __name__ == "__main__":
    max_epoch = 50
    training(max_epoch)
    testing(max_epoch)
