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
import dataloader as dt
import torchshow as ts
import timm

check_mode = True
generate_mode = False


def pgd_attack(model, images, labels, eps=0.3, alpha=2 / 255, iters=40):
	images = images.to(device)
	labels = labels.to(device)
	loss = nn.CrossEntropyLoss()

	ori_images = images.data

	for i in range(iters):
		images.requires_grad = True
		outputs = model(images)

		model.zero_grad()
		cost = loss(outputs, labels).to(device)
		cost.backward()

		adv_images = images + alpha * images.grad.sign()
		eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
		images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

	return images


device = torch.device("cuda")
#model = models.inception_v3(pretrained=True).to(device)
model = models.resnet50(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
vit = False

dloader_clean = dt.get_loaders_v2('data/test_data_1/sprt-test-set-clean-pt-224/test_data.pt', 'data/test_data_1/sprt-test-set-clean-pt-224/test_labels.pt')
dloader_pgd = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt')

if check_mode:
	model.eval()
	correct = 0
	total = 0
	for i, (images, labels) in enumerate(dloader_pgd):
		print(str(i) + " of " + str(len(dloader_pgd)))
		images = images.to(device)
		labels = labels.to(device)

		if vit:
			transform = transforms.Resize((224, 224))
			images = transform(images)

		outputs = model(images)

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))  # clean inception acc: 76.86 & adv inception acc: 0.020
																				# clean resnet acc: 74.50 & adv resnet acc: 0.00

if generate_mode:
	model.eval()
	correct = 0
	total = 0
	for i, (images, labels) in enumerate(dloader_clean):
		print(str(i) + " of " + str(len(dloader_clean)))
		labels = labels.type(torch.long)
		images = pgd_attack(model, images, labels)
		labels = labels.to(device)
		outputs = model(images)

		torch.save(images, 'chunk/tensor' + str(i) + '.pt')

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))