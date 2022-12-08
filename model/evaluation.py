
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import VGG_binary_model
from config.label_index import label_index


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


if __name__ == '__main__':
    dir_X = 'Au_herringbone/training_data/binary_train_X_augmentation.npy'
    dir_y = 'Au_herringbone/training_data/binary_train_y_augmentation.npy'

    data_X = np.load(dir_X)
    data_y = np.load(dir_y)
    print('data_X.shape', data_X.shape)
    print('data_y.shape', data_y.shape)
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3, random_state=22, shuffle=True)
    testset = Data(test_X, test_y)
    test_loader = DataLoader(testset, batch_size=64)

    model_path = r'Au_herringbone/model/trained_model/VGG_binary_new_augmentation.pt'
    checkpoint = torch.load(model_path)
    model = VGG_binary_model()
    model.load_state_dict(checkpoint)

    f1 = F1Score(num_classes=2)

    with torch.no_grad():
        model.eval()

        for X, y in tqdm(test_loader):
            pred_y = model(X)
            f1(pred_y, y)

        f1_score = f1.compute()

        print('F1 score:', f1_score)