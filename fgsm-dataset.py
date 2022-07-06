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
from torchvision.utils import save_image

import dataloader as dt
import torchshow as ts
import cv2
from PIL import Image

lbl_dict = dict([(0, 0), (1, 217), (2, 482), (3, 491), (4, 497), (5, 566), (6, 569), (7, 571), (8, 574), (9, 701)])


def fgsm_attack(model, loss, images, labels, eps):
	images = images.to(device)
	labels = labels.to(device)
	images.requires_grad = True

	outputs = model(images)

	model.zero_grad()
	cost = loss(outputs, labels).to(device)
	cost.backward()

	attack_images = images + eps * images.grad.sign()
	attack_images = torch.clamp(attack_images, 0, 1)

	return attack_images


if __name__ == '__main__':
	valdir = 'C:/Users/furkan/Desktop/projects/combine-attack/data/imagenette2-320-tiny50/val'
	eps = 0.001  # 0.007
	use_cuda = True

	device = torch.device("cuda" if use_cuda else "cpu")
	model = models.resnet50(pretrained=True).to(device)

	dloader = dt.get_loaders(valdir)

	print("True Image & Predicted Label")

	model.eval()

	correct = 0
	total = 0

	model2 = models.resnet50(pretrained=True).to(device)
	correct2 = 0
	total2 = 0
	model2.eval()

	for i, (images, labels) in enumerate(dloader):
		print(str(i) + " of " + str(len(dloader)))
		labels = torch.ones(10) * lbl_dict[int(labels[0])]
		labels = labels.type(torch.long)

		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

		outputs2 = model2(images)
		_2, pre2 = torch.max(outputs2.data, 1)
		total2 += 1
		correct2 += (pre2 == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))
	print('Accuracy of test text: %f %%' % (100 * float(correct2) / (total2 * 10)))

	###

	print("Attack Image & Predicted Label")
	model.eval()

	correct = 0
	total = 0
	loss = nn.CrossEntropyLoss()

	correct2 = 0
	total2 = 0

	for i, (images, labels) in enumerate(dloader):
		print(str(i) + " of " + str(len(dloader)))
		images = fgsm_attack(model, loss, images, labels, eps).to(device)
		labels = labels.to(device)
		outputs = model(images)

		_, pre = torch.max(outputs.data, 1)
		#for image in images:
		#	ts.save(image)

		#for j in range(10):
		#	name = str((i * 10) + j)
		#	save_image(images[j], 'deneme/' + name + '.png')

		#image = images[1].permute(1, 2, 0).cpu().detach().numpy()
		#image = cv2.convertScaleAbs(image, alpha=(255.0))
		#cv2.imwrite("frame.jpg", image)

		#torch.save(images, 'tensor' + str(i) + '.pt')

		total += 1
		correct += (pre == labels).sum()

		outputs2 = model2(images)
		_2, pre2 = torch.max(outputs2.data, 1)
		total2 += 1
		correct2 += (pre2 == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))
	print('Accuracy of test text: %f %%' % (100 * float(correct2) / (total2 * 10)))