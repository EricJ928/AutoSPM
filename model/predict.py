import numpy as np
import torch
from model import VGG_binary_model
from matplotlib import pyplot as plt
from config.label_index import label_index

model_path = r'Au_herringbone/model/trained_model/VGG_binary_new_augmentation.pt'
checkpoint = torch.load(model_path)
model = VGG_binary_model()
model.load_state_dict(checkpoint)
model.eval()

# test_file = r'Au_herringbone/Au_example/training_test_set_binary/img_good/Au caliib_1007_007.npy'
test_file = r'Au_herringbone/Au_example/training_test_set_binary/img_bad/Au caliib_1006_001.npy'
query = np.load(test_file)
# print(query.shape)

tensor_query = torch.FloatTensor(query).reshape(1, 1, 64, 64)
# print(tensor_query.shape)
output = model.forward(tensor_query).detach()
print('output:', output)
# probs = torch.nn.functional.softmax(output, dim=1)
# print('probs:', probs)
conf, classes = torch.max(output, 1)
print('conf:', conf)
print('classes:', classes)
index = classes.item()

index_to_label_binary_dict = label_index['index_to_label_binary_dict']
print('result:', index_to_label_binary_dict[index])

plt.figure()
plt.imshow(query, origin='lower')
plt.title(f'Model Prediction: {index_to_label_binary_dict[index]}')
plt.axis('off')
plt.show()