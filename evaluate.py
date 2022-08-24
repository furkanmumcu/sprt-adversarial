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
lbl_dict = dict([(0, 0), (1, 217), (2, 482), (3, 491), (4, 497), (5, 566), (6, 569), (7, 571), (8, 574), (9, 701)])
dct = dataloader.get_test_dict(test_dict)

if __name__ == '__main__':
	fgsm4 = 'C:/Users/furkan/Desktop/projects/combine-attack/data/adv-fgsm4'
	pfool = 'C:/Users/furkan/Desktop/projects/combine-attack/data/adv-patchfool'
	use_cuda = True

	test_data = 'C:/Users/furkan/Desktop/projects/combine-attack/data/test_data_1/sprt-test-set'

	device = torch.device("cuda" if use_cuda else "cpu")
	model = models.resnet50(pretrained=True).to(device)
	#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
	#model = deit_small_patch16_224(pretrained=True).to(device)
	deit = False

	dloader = dt.get_loaders(test_data)

	print("True Image & Predicted Label")

	model.eval()

	correct = 0
	total = 0

	for i, (images, labels) in enumerate(dloader):
		print(str(i) + " of " + str(len(dloader)))

		#labels = torch.ones(10) * lbl_dict[int(labels[0])]
		labels = torch.ones(10) * dct[str(int(labels[0]))]
		labels = labels.type(torch.long)

		#images2 = torch.load('data/adv-fgsm-pt/tensor' + str(i) + '.pt')
		#images3 = torch.load('data/adv-patchfool-pt/tensor' + str(i) + '.pt')

		images = images.to(device)
		#images2 = images2.to(device)
		#images3 = images3.to(device)
		labels = labels.to(device)

		outputs = model(images)
		if deit:
			outputs = outputs[0]

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))