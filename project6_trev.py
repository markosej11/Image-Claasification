import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import os
import pandas as pd
import io
import cv2


class AnimalDataset(Dataset):
    """
    Animal dataset.

    References
    ----------
    https://www.youtube.com/watch?v=ZoZHd0Zm3RY
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_name = self.annotations.iloc[idx, 0]
        image = cv2.imread(img_name)
        animalType = torch.tensor(int(self.annotations.iloc[idx, 1]))
        # animalType = animalType.reshape(-1, 2)

        if self.transform:
            image = self.transform(image)

        return image, animalType


class ConvNet(nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        # 28x28x1 => 28x28x8
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(28-1) - 28 + 3) / 2 = 1
        # 28x28x8 => 14x14x8
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)  # (2(14-1) - 28 + 2) = 0
        # 14x14x8 => 14x14x16
        self.conv_2 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(14-1) - 14 + 3) / 2 = 1
        # 14x14x16 => 7x7x16
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)  # (2(7-1) - 14 + 2) = 0

        self.linear_1 = torch.nn.Linear(7 * 7 * 16, num_classes)

    def forward(self, x):
        out = self.conv_1(x)
        out = func.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = func.relu(out)
        out = self.pool_2(out)

        logits = self.linear_1(out.view(-1, 7 * 7 * 16))
        probas = func.softmax(logits, dim=1)
        return logits, probas


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imShow(img):
    img = img / 2 + 0.5     # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def findImages(directory):
    source_tree = list(os.walk(directory))
    root, subdir, filenames = source_tree[0]
    return [("%s/%s" % (root, filename)) for filename in filenames]


def main():
    """
    Classification implementation using PyTorch.

    References
    ----------
    Loading data:  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    num_epoch = 10

    # Read training data and categories.  Also read test data.
    cwd = os.getcwd().replace("\\", "/")
    trainDir = "%s/training" % cwd
    testDir = "%s/testing" % cwd
    csvFileName_train = "%s/categories.csv" % cwd
    csvFileName_test = "%s/test_images.csv" % cwd
    trainFileNames = findImages(trainDir)
    testFileNames = findImages(testDir)
    categoryCSV = io.open(csvFileName_train, mode="wt")
    for name in trainFileNames:
        category = name.split("/")[-1].split(".")[0]
        categoryCSV.write("%s,%d\n" % (name, category == "dog"))
    testCSV = io.open(csvFileName_test, mode="wt")
    for name in testFileNames:
        testCSV.write("%s\n" % name)

    # Create training and testing sets of images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainingSet = AnimalDataset(csvFileName_train, trainDir, transform=transform)
    trainingLoader = DataLoader(trainingSet, batch_size=4, shuffle=True, num_workers=2)
    testSet = AnimalDataset(csvFileName_test, testDir, transform=transform)
    testLoader = DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)
    exit(0)

    classes = ("dog", "cat")

    # Get some random training images
    dataIterator = iter(trainingLoader)
    images, labels = next(dataIterator)
    print(classes, len(classes))
    # show images
    imShow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

    # Create neural net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ConvNet(2)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()

    # Save the net (just to show how it's done)
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

    # Training
    for epoch in range(num_epoch):
        losses = []
        for index, (data, targets) in enumerate(trainingLoader):
            inputs, labels = data[0].to(device), data[1].to(device)
            print(inputs, labels)
    exit(0)

    # Testing
    dataIterator = iter(testLoader)
    images, labels = next(dataIterator)
    # print images
    imShow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

    # Check output of testing for a few images
    outputs = net(images)
    predicted = torch.max(outputs, 1)[1]
    print('Predicted: ', ' '.join('%5s' % classes[int(predicted[j])]
                                  for j in range(4)))

    # See results for entire data set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    # See sub-par results
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    main()
