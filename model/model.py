
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy


class Au_model:
    """
    Model for training and predicting. The input training and prediect set need to be preprocessed
    (subtract 2d plane and normalized).

    Parameters:     train_X : rank3 numpy array
                        Training set input. shape[0] represents samples. shape[1]+[2] represnts
                        image data matrix.

                    train_y : rank1 numpy array
                        Training set index from labels.
                        
                    query : rank2 or rank3 numpy array
                        predict data input. Can be a single image or a series of images.
    """

    def __init__(self, model_name, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        
        if model_name == 'VGG_binary':
            self.classifier = VGG_binary_model()
        else:
            print('No model found!')
            return
        print("--> NN Architecture:", self.classifier)

        self.losses = []
        self.accuracy = []
    

    def training(self):
        
        dataset = Data(self.train_X, self.train_y)
        loader = DataLoader(dataset, batch_size=64)

        learning_rate = 0.001
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

        loss_epoch = []
        self.epochs = []
        self.losses = []
        self.accuracy = []
        metric = Accuracy()

        epoch_num = 40
        for epoch in range(epoch_num):
            count = 0
            loss_cache = 0
            
            for X,y in tqdm(loader, desc=f'Epoch {epoch+1}/{epoch_num}'):
                self.classifier.zero_grad()
                pred_y = self.classifier(X)
                loss = loss_function(pred_y, y)
                loss_epoch.append(loss.item())
                loss_cache += loss.item()
                count += 1
                acc = metric(pred_y, y)
                loss.backward()
                optimizer.step()
            self.epochs.append(epoch+1)
            self.losses.append(loss_cache/count)
            acc = metric.compute()
            self.accuracy.append(acc.numpy())
            metric.reset()
            print(f'--> epoch: {epoch+1}/{epoch_num}  --  loss: {loss_cache/count}  --  accuracy: {acc.numpy()}')
    

    def training_cross_validation(self, test_X, test_y):
        
        train_dataset = Data(self.train_X, self.train_y)
        train_loader = DataLoader(train_dataset, batch_size=64)
        test_dataset = Data(test_X, test_y)
        test_loader = DataLoader(test_dataset, batch_size=64)

        learning_rate = 0.001
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

        self.epochs = []

        self.epoch_losses_train = []
        self.epoch_accuracy_train = []
        # self.epoch_losses_test = []
        self.epoch_accuracy_test = []

        self.batch_losses_train = []
        self.batch_accuracy_train = []
        # self.batch_losses_test = []
        # self.batch_accuracy_test = []

        metric_train = Accuracy()
        metric_test = Accuracy(threshold=0.5)

        epoch_num = 40
        for epoch in range(epoch_num):
            count = 0
            loss_cache = 0

            # Training
            for X,y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epoch_num}'):
                self.classifier.zero_grad()
                pred_y = self.classifier(X)
                loss = loss_function(pred_y, y)
                loss_item = loss.item()
                loss_cache += loss_item
                count += 1
                acc = metric_train(pred_y, y)
                loss.backward()
                optimizer.step()

                self.batch_losses_train.append(loss_item)
                self.batch_accuracy_train.append(acc.numpy())

            self.epochs.append(epoch+1)
            self.epoch_losses_train.append(loss_cache/count)

            acc = metric_train.compute()
            self.epoch_accuracy_train.append(acc.numpy())
            metric_train.reset()

            # Test
            for X,y in test_loader:
                pred_y = self.classifier(X)
                acc_test = metric_test(pred_y, y)

            acc_test = metric_test.compute()
            self.epoch_accuracy_test.append(acc_test.numpy())
            metric_test.reset()

            print(f'--> epoch: {epoch+1}/{epoch_num} -- training loss: {loss_cache/count} -- training accuracy: {acc.numpy()} -- test accuracy: {acc_test.numpy()}')


    def test(self, test_X, test_y, model_name, version):
        model_path = f"Au_herringbone/model/trained_model/{model_name}_{version}.pth"
        checkpoint = torch.load(model_path)
        self.classifier.load_state_dict(checkpoint)

        dataset = Data(test_X, test_y)
        loader = DataLoader(dataset, batch_size=64)

        metric = Accuracy(threshold=0.5)
        for X,y in loader:
            pred_y = self.classifier(X)
            acc = metric(pred_y, y)
        acc = metric.compute()
        return acc.numpy()


    def predict(self, query, model_name, version):
        model_path = f"Au_herringbone/model/trained_model/{model_name}_{version}.pth"
        checkpoint = torch.load(model_path)
        self.classifier.load_state_dict(checkpoint)
        tensor_query = torch.FloatTensor(query).reshape(1, 64, 64)
        predict_proba = self.classifier(tensor_query).detach().numpy()
        return predict_proba


class VGG_binary_model(nn.Module):
    """
    Neural network model for modified VGG classifier.
    """
    def __init__(self):
        super(VGG_binary_model, self).__init__()
        self.conv11 = nn.Conv2d(1, 16, 3, padding="same")
        self.conv12 = nn.Conv2d(16, 16, 3, padding="same")
        self.conv13 = nn.Conv2d(16, 16, 3, padding="same")

        self.conv21 = nn.Conv2d(16, 32, 3, padding="same")
        self.conv22 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv23 = nn.Conv2d(32, 32, 3, padding="same")

        self.conv31 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv32 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv33 = nn.Conv2d(64, 64, 3, padding="same")


        self.fc1 = nn.Linear(8*8*64, 8*8*16)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(8*8*16, 2)


    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.max_pool2d(x, 2)


        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class Data(Dataset):
    """
    Data set preparation for training.
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        # self.y = torch.FloatTensor(y)
        self.len = self.X.shape[0]


    def __getitem__(self, index):
        return self.X[index], self.y[index]


    def __len__(self):
        return self.len
