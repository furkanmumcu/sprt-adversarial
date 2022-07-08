import numpy as np
import json
from itertools import cycle

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
import urllib
import timm
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

lbl_dict = dict([(0, 0), (1, 217), (2, 482), (3, 491), (4, 497), (5, 566), (6, 569), (7, 571), (8, 574), (9, 701)])
batch_size = 10
upper_bound = 5
lower_bound = -5


# load clean dataset
clean_path = 'C:/Users/furkan/Desktop/projects/combine-attack/data/imagenette2-320-tiny50/val'
loader_clean = dt.get_loaders(clean_path)


# load first adv dataset
fgsm_path = 'C:/Users/furkan/Desktop/projects/combine-attack/data/adv-fgsm4'
loader_fgsm = dt.get_loaders(fgsm_path)

# load second adv dataset
patchfool_path = 'C:/Users/furkan/Desktop/projects/combine-attack/data/adv-patchfool'
loader_patchfool = dt.get_loaders(patchfool_path)


# define surrogate model
device = torch.device("cuda")

model = models.resnet50(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
#model = models.inception_v3(pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)
deit = False

model.eval()  # ?


if __name__ == '__main__':
	s = 0
	summ = 0

	# get adv dataset's accuracy and combine them
	for i, ((X1, y1), (X2, y2)) in enumerate(zip(loader_fgsm, cycle(loader_patchfool))):
		#print(str(i) + " of " + str(len(loader_patchfool)))

		y1 = torch.ones(10) * lbl_dict[int(y1[0])]
		y1 = y1.type(torch.long)
		y2 = torch.ones(10) * lbl_dict[int(y2[0])]
		y2 = y2.type(torch.long)

		X1_2 = torch.load('data/adv-fgsm-pt/tensor' + str(i) + '.pt')
		X1_2 = X1_2.to(device)
		X2_2 = torch.load('data/adv-patchfool-pt/tensor' + str(i) + '.pt')
		X2_2 = X2_2.to(X2_2)

		X1 = X1.to(device)
		y1 = y1.to(device)
		X2 = X2.to(device)
		y2 = y2.to(device)

		model.eval()
		with torch.no_grad():
			out = model(X1_2)

		if deit:
			out = out[0]

		_, pre = torch.max(out.data, 1)

		probabilities1 = torch.nn.functional.softmax(out, dim=1)
		top5_prob1, top5_catid1 = torch.topk(probabilities1, 1000)

		# calculate delta_p_1
		deltas1 = []
		for j in range(len(y1)):
			true_label = y1[j].item()
			true_label_index = (top5_catid1[j] == true_label).nonzero(as_tuple=False).item()
			prob = top5_prob1[j][true_label_index].item()
			#delta = true_probs[(i*batch_size) + j] - prob #+ offset[(i*batch_size) + j]
			delta = 1 - prob
			deltas1.append(delta)

		deltas1 = np.asarray(deltas1)
		#print(deltas1.shape)

		#model = models.resnet50(pretrained=True).to(device)
		model.eval()
		with torch.no_grad():
			out = model(X2_2)

		if deit:
			out = out[0]

		_, pre = torch.max(out.data, 1)

		probabilities2 = torch.nn.functional.softmax(out, dim=1)
		top5_prob2, top5_catid2 = torch.topk(probabilities2, 1000)

		# calculate delta_p_2
		deltas2 = []
		for j in range(len(y2)):
			true_label = y2[j].item()
			true_label_index = (top5_catid2[j] == true_label).nonzero(as_tuple=False).item()
			prob = top5_prob2[j][true_label_index].item()
			#delta = true_probs[(i * batch_size) + j] - prob #+ offset[(i * batch_size) + j]
			delta = 1 - prob
			deltas2.append(delta)

		deltas2 = np.asarray(deltas2)
		#print(deltas2.shape)

		deltas = np.log(deltas1 / deltas2)

		for k in range(deltas.shape[0]):
			if deltas[k] < 0:
				s = s - 1
			else:
				s = s + 1

		summ = deltas.sum() + summ

		print(s)
		print(summ)
		'''
		if s <= lower_bound:
			print('this is a Transformer')
			break
		elif s >= upper_bound:
			print('this is a CNN')
			break
		else:
			print('going to next batch')
		'''