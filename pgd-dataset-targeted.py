import numpy as np
import json
from random import randint

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


def random_exclude(*exclude):
	exclude = set(exclude)
	randInt = randint(0, 999)
	return random_exclude() if randInt in exclude else randInt


def pgd_attack(model, images, labels, eps=0.3, alpha=2 / 255, iters=40):
	images = images.to(device)
	labels = labels.to(device)
	loss = nn.CrossEntropyLoss()

	ori_images = images.data

	target_labels = []
	for label in labels:
		target_label = random_exclude(label)
		target_labels.append(target_label)
	target_labels = np.asarray(target_labels)
	target_labels = torch.from_numpy(target_labels)
	target_labels = target_labels.to(device)
	target_labels = target_labels.type(torch.long)

	for i in range(iters):
		images.requires_grad = True
		outputs = model(images)
		if deit:
			outputs = outputs[0]

		model.zero_grad()
		#cost = loss(outputs, labels).to(device)
		cost = -loss(outputs, target_labels)

		cost.backward()

		adv_images = images + alpha * images.grad.sign()
		eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
		images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

	return images, target_labels


device = torch.device("cuda")
#model = models.inception_v3(pretrained=True).to(device)
#model = models.resnet50(pretrained=True).to(device)
model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)

is_Transform = True
deit = False

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

		if is_Transform:
			transform = transforms.Resize((224, 224))
			images = transform(images)

		outputs = model(images)

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (
				100 * float(correct) / (total * 10)))

if generate_mode:
	model.eval()
	correct = 0
	total = 0
	target_label_success = 0
	for i, (images, labels) in enumerate(dloader_clean):
		if i >= 0:
			print(str(i) + " of " + str(len(dloader_clean)))
			labels = labels.type(torch.long)
			result = pgd_attack(model, images, labels)
			t_labels = result[1]
			images = result[0]
			labels = labels.to(device)
			outputs = model(images)
			if deit:
				outputs = outputs[0]

			torch.save(images, 'chunk/tensor' + str(i) + '.pt')
			torch.save(t_labels, 'chunk_tlabel/tensor' + str(i) + '.pt')

			_, pre = torch.max(outputs.data, 1)

			total += 1
			target_label_success += (pre == t_labels).sum()
			correct += (pre == labels).sum()
		else:
			print('skipped')

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total * 10)))
	print('target label success: %f %%' % (100 * float(target_label_success) / (total * 10)))
