import numpy as np
import json
import copy

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
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

check_mode = False
generate_mode = True


def pgd_attack(model1, model2, images1, labels1, eps=0.3, alpha=2 / 255, iters=40):
	org_images = copy.deepcopy(images1)
	org_images = org_images.to(device)

	images2 = copy.deepcopy(images1)
	labels2 = copy.deepcopy(labels1)

	images1 = images1.to(device)
	images2 = images2.to(device)
	labels1 = labels1.to(device)
	labels2 = labels2.to(device)
	loss1 = nn.CrossEntropyLoss()
	loss2 = nn.CrossEntropyLoss()

	ori_images = org_images.data

	for i in range(iters):
		# grad 1
		images1.requires_grad = True
		outputs = model1(images1)
		if deit:
			outputs = outputs[0]

		model1.zero_grad()
		cost = loss1(outputs, labels1).to(device)
		cost.backward()

		# grad 2
		images2.requires_grad = True
		outputs = model2(images2)
		if deit:
			outputs = outputs[0]

		model2.zero_grad()
		cost = loss2(outputs, labels2).to(device)
		cost.backward()

		#

		grad1 = images1.grad.sign()
		grad2 = images2.grad.sign()

		grad = (grad1 + grad2) / 2

		adv_images = org_images + alpha * grad
		eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
		org_images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

	return org_images


device = torch.device("cuda")
model1 = models.inception_v3(pretrained=True).to(device)
model2 = models.resnet50(pretrained=True).to(device)

#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)

is_Transform = False
deit = False

dloader_clean = dt.get_loaders_v2('data/test_data_1/sprt-test-set-clean-pt-224/test_data.pt', 'data/test_data_1/sprt-test-set-clean-pt-224/test_labels.pt')


if generate_mode:
	model1.eval()
	model2.eval()

	correct1 = 0
	total1 = 0
	correct2 = 0
	total2 = 0
	for i, (images, labels) in enumerate(dloader_clean):
		print(str(i) + " of " + str(len(dloader_clean)))
		labels = labels.type(torch.long)
		images = pgd_attack(model1, model2, images, labels)
		labels = labels.to(device)

		outputs1 = model1(images)
		if deit:
			outputs1 = outputs1[0]

		outputs2 = model2(images)
		if deit:
			outputs2 = outputs2[0]

		torch.save(images, 'chunk/tensor' + str(i) + '.pt')

		_, pre = torch.max(outputs1.data, 1)

		total1 += 1
		correct1 += (pre == labels).sum()

		_, pre = torch.max(outputs2.data, 1)

		total2 += 1
		correct2 += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct1) / (total1*10)))
	print('Accuracy of test text: %f %%' % (100 * float(correct2) / (total2 * 10)))