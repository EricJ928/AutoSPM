
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import sys
sys.path.append("C:\\Users\\Eric JIA\\OneDrive - Nanyang Technological University\\Eric\\Projects\\AI_STM")

from model import Au_model, Data
from utils import data_preprocess as dp


def train_model(train_X, train_y, model_name, version):
    model = Au_model(model_name=model_name, train_X=train_X, train_y=train_y)
    model.training()
    torch.save(model.classifier.state_dict(), f"Au_herringbone/model/trained_model/{model_name}_{version}.pt")
    # print(model.losses)
    # print(model.accuracy)
    data_df = pd.DataFrame(model.epochs, columns=['epoch'])
    data_df['loss'] = model.losses
    data_df['accuracy'] = model.accuracy

    data_df.to_csv(f'model/trained_model/{model_name}_{version}_accuracy.csv', index=False)


def train_test_model(train_X, train_y, test_X, test_y, model_name, version):
    model = Au_model(model_name=model_name, train_X=train_X, train_y=train_y)
    model.training_cross_validation(test_X=test_X, test_y=test_y)
    torch.save(model.classifier.state_dict(), f"Au_herringbone/model/trained_model/{model_name}_{version}.pt")
    
    epoch_df = pd.DataFrame(model.epochs, columns=['epoch'])
    epoch_df['training_loss'] = model.epoch_losses_train
    epoch_df['training_accuracy'] = model.epoch_accuracy_train
    epoch_df['test_accuracy'] = model.epoch_accuracy_test
    epoch_df.to_csv(f'model/trained_model/{model_name}_{version}_epoch_accuracy.csv', index=False)

    batch_df = pd.DataFrame(model.epochs, columns=['epoch'])
    batch_df['training_loss'] = model.epoch_losses_train
    batch_df['training_accuracy'] = model.epoch_accuracy_train
    batch_df.to_csv(f'model/trained_model/{model_name}_{version}_batch_accuracy.csv', index=False)


def test_model(query, test_y, model_name, version):
    model = Au_model(model_name=model_name, train_X=None, train_y=None)
    checkpoint = torch.load(f"Au_herringbone/model/trained_model/{model_name}_{version}.pt")
    model.classifier.load_state_dict(checkpoint)

    test_accuracy = model.test(test_X=query, test_y=test_y, model_name=model_name, version=version)
    print(test_accuracy)


if __name__ == '__main__':
    # dir_X = 'training_data/train_X.npy'
    # dir_y = 'training_data/train_y.npy'
    # dir_X = 'training_data/train_X_augmentation.npy'
    # dir_y = 'training_data/train_y_augmentation.npy'
    # dir_X = 'training_data/binary_train_X.npy'
    # dir_y = 'training_data/binary_train_y.npy'
    dir_X = 'Au_herringbone/training_data/binary_train_X_augmentation.npy'
    dir_y = 'Au_herringbone/training_data/binary_train_y_augmentation.npy'

    data_X = np.load(dir_X)
    data_y = np.load(dir_y)
    print('data_X.shape', data_X.shape)
    print('data_y.shape', data_y.shape)
    # data_y = data_y.reshape((data_y.shape[0], 1))
    # data_y = data_y.astype(int)
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3, random_state=22, shuffle=True)

    # print(train_y)

    # train_model(train_X=train_X, train_y=train_y, model_name='VGG_binary', version='augmentation')
    # train_test_model(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, model_name='VGG_binary', version='augmentation')
    # test_model(query=test_X, test_y=test_y, model_name='VGG_binary', version='augmentation')

    # train_model(train_X=train_X, train_y=train_y, model_name='VGG_binary', version='augmentation')
    # train_test_model(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, model_name='VGG_binary', version='new')
    # train_test_model(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, model_name='VGG_binary', version='new_augmentation')
    # test_model(query=test_X, test_y=test_y, model_name='VGG_binary', version='augmentation')

    # metric = Accuracy(threshold=0.5)
    # acc = metric(torch.FloatTensor(y_pred), torch.LongTensor(test_y))
    # print(acc)

    # print(y_pred)
    # y_pred_binary = y_pred > 0.5
    # print(y_pred_binary)

    # count = 0
    # for i in range(y_pred_binary.shape[0]):
    #     if y_pred_binary[i] == test_y[i]:
    #         count += 1
    # print(count/y_pred_binary.shape[0])
