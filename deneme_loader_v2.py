import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import dataloader
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

import dataloader as dt
import torchshow as ts
import timm


test_dict = 'C:/Users/furkan/Desktop/projects/combine-attack/data/test_data_1/test_dict.txt'
dct = dataloader.get_test_dict(test_dict)

if __name__ == '__main__':
	use_cuda = True

	device = torch.device("cuda" if use_cuda else "cpu")
	model = models.resnet50(pretrained=True).to(device)
	#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
	#model = deit_small_patch16_224(pretrained=True).to(device)
	deit = False

	dloader_adv_fgsm = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-1/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-1/test_labels.pt')
	dloader_clean = dt.get_loaders_v2('data/test_data_1/sprt-test-set-clean-pt/test_data.pt', 'data/test_data_1/sprt-test-set-clean-pt/test_labels.pt')

	print("True Image & Predicted Label")

	model.eval()

	correct = 0
	total = 0

	for i, (images, labels) in enumerate(dloader_clean):
		print(str(i) + " of " + str(len(dloader_clean)))

		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		if deit:
			outputs = outputs[0]

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))