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

check_mode = False
generate_mode = True


def cw_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
	images = images.to(device)
	labels = labels.to(device)

	# Define f-function
	def f(x):

		outputs = model(x)
		one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

		i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
		j = torch.masked_select(outputs, one_hot_labels.bool())

		# If targeted, optimize for making the other class most likely
		if targeted:
			return torch.clamp(i - j, min=-kappa)

		# If untargeted, optimize for making the other class most likely
		else:
			return torch.clamp(j - i, min=-kappa)

	w = torch.zeros_like(images, requires_grad=True).to(device)

	optimizer = optim.Adam([w], lr=learning_rate)

	prev = 1e10

	for step in range(max_iter):

		a = 1 / 2 * (nn.Tanh()(w) + 1)

		loss1 = nn.MSELoss(reduction='sum')(a, images)
		loss2 = torch.sum(c * f(a))

		cost = loss1 + loss2

		optimizer.zero_grad()
		cost.backward()
		optimizer.step()

		# Early Stop when loss does not converge.
		if step % (max_iter // 10) == 0:
			if cost > prev:
				print('Attack Stopped due to CONVERGENCE....')
				return a
			prev = cost

		print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')

	attack_images = 1 / 2 * (nn.Tanh()(w) + 1)

	return attack_images


device = torch.device("cuda")
model = models.inception_v3(pretrained=True).to(device)
#model = models.resnet50(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
vit = False

dloader_clean = dt.get_loaders_v2('data/test_data_1/sprt-test-set-clean-pt-299/test_data.pt', 'data/test_data_1/sprt-test-set-clean-pt-299/test_labels.pt')
dloader_pgd = None

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

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))  # clean inception acc: 76.86

if generate_mode:
	model.eval()
	correct = 0
	total = 0
	for i, (images, labels) in enumerate(dloader_clean):
		print(str(i) + " of " + str(len(dloader_clean)))
		labels = labels.type(torch.long)
		images = cw_l2_attack(model, images, labels)
		labels = labels.to(device)
		outputs = model(images)

		torch.save(images, 'chunk/tensor' + str(i) + '.pt')

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))  # adv inception acc: 0.020