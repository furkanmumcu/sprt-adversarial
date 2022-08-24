import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
import numpy as np
from os import path
import os
import timm
import urllib
import matplotlib.pyplot as plt
import torchshow as ts
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import json

mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#dics needed for imagenette
lbl_dict_names = dict(
	n01440764='tench',  # 0
	n02102040='English springer',  # 217
	n02979186='cassette player',  # 482
	n03000684='chain saw',  # 491
	n03028079='church',  # 497
	n03394916='French horn',  # 566
	n03417042='garbage truck',  # 569
	n03425413='gas pump',  # 571
	n03445777='golf ball',  # 574
	n03888257='parachute'  # 701
)

lbl_dict_ids = dict(
	n01440764=0,  # 0
	n02102040=217,  # 217
	n02979186=482,  # 482
	n03000684=491,  # 491
	n03028079=497,  # 497
	n03394916=566,  # 566
	n03417042=569,  # 569
	n03425413=571,  # 571
	n03445777=574,  # 574
	n03888257=701  # 701
)

lbl_dict = dict([(0, 0), (1, 217), (2, 482), (3, 491), (4, 497), (5, 566), (6, 569), (7, 571), (8, 574), (9, 701)])


#dics needed for our testing set
def get_test_dict(dir):
	with open(dir) as json_data:
		index = json.load(json_data)
	return index


class CustomImageFolder(ImageFolder):

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (sample, target) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		#print(11111)
		return sample, target


def get_loaders(dir, size=224):
	#valdir = path.join(args.data_dir, 'val')
	valdir = 'C:/Users/furkan/Desktop/projects/combine-attack/data/imagenette2-320-tiny50/val'
	val_dataset = CustomImageFolder(dir,
									transforms.Compose([transforms.Resize(size),
														transforms.CenterCrop(size),
														transforms.ToTensor(),
														transforms.Normalize(mean=mu, std=std)
														]))
	val_dataset.__getitem__(1)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False,
											 num_workers=0, pin_memory=True)
	return val_loader


class DataFromFile(Dataset):
	def __init__(self, data_path, label_path):

		#self.data = pd.read_csv(csv_path)
		self.data = torch.load(data_path)

		#self.labels = np.asarray(self.data.iloc[:, 0])
		self.labels = torch.load(label_path)

	def __getitem__(self, index):
		single_image_data = self.data[index]
		single_image_label = self.labels[index]

		return single_image_data, single_image_label

	def __len__(self):
		return len(self.data)


def get_loaders_v2(data_path, label_path, batch=10, shuffle=False):
	dataset = DataFromFile(data_path, label_path)

	dataset.__getitem__(1)
	val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=False)
	return val_loader


'''
if __name__ == '__main__':
	device = torch.device("cuda")
	model = models.resnet50(pretrained=True).to(device)
	model.eval()
	correct = 0
	total = 0

	data_path = 'data/adv-fgsm-pt-complete/data.pt'
	label_path = 'data/adv-fgsm-pt-complete/labels.pt'
	loader = get_loaders_v2(data_path, label_path)

	for i, (images, labels) in enumerate(loader):
		print(str(i) + " of " + str(len(loader)))

		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total * 10)))
'''

# get clean dataset tensor file
if __name__ == '__main__':
	dct = get_test_dict('C:/Users/furkan/Desktop/projects/combine-attack/data/test_data_1/test_dict.txt')
	test_data = 'C:/Users/furkan/Desktop/projects/combine-attack/data/test_data_1/sprt-test-set'
	loader = get_loaders(test_data, size=299)

	data_clt = torch.tensor([])
	labels_clt = torch.tensor([])
	for i, (images, labels) in enumerate(loader):
		print(str(i) + " of " + str(len(loader)))

		labels = torch.ones(10) * dct[str(int(labels[0]))]

		data_clt = torch.cat((data_clt, images), 0)
		labels_clt = torch.cat((labels_clt, labels), 0)

	print(data_clt.shape)
	print(labels_clt.shape)

	torch.save(data_clt, 'test_data.pt')
	torch.save(labels_clt, 'test_labels.pt')