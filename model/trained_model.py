import numpy as np
import torch
from model import Au_model, Data


def pridict_model(query, model_name, version):
    model = Au_model(model_name=model_name, train_X=None, train_y=None)

    predict_proba = model.predict(query, model_name, version)
    print(predict_proba)


# model_path = r'Au_herringbone/model/trained_model/VGG_binary_new_augmentation.pth'
# model = torch.load(model_path)

test_file = r'Au_herringbone/Au_example/training_test_set_binary/img_good/Au caliib_1007_007.npy'
data = np.load(test_file)
print(data.shape)

pridict_model(query=data, model_name='VGG_binary', version='new_augmentation')