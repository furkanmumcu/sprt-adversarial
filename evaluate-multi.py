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
import utils
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

import dataloader as dt
import torchshow as ts
import timm

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

#loader_fgsm_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-deits/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-deits/test_labels.pt')
#loader_pgd_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/deit-s/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/deit-s/test_labels.pt')
#loader_pfool_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-s/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-s/test_labels.pt')
#loader_pna_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna-deits/test_data.pt', 'data/test_data_1/sprt-test-set-pna-deits/test_labels.pt')

loader_fgsm_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-inception/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-inception/test_labels.pt')
#loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt')
#loader_cw_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt')
#loader_pna_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna-inception/test_data.pt', 'data/test_data_1/sprt-test-set-pna-inception/test_labels.pt')


def evaluate(model, deit, is_transform):
	model.eval()

	correct = 0
	total = 0

	for i, (images, labels) in enumerate(loader_fgsm_inception):
		print(str(i) + " of " + str(len(loader_fgsm_inception)))

		images = images.to(device)
		labels = labels.to(device)

		if is_transform:
			transform = transforms.Resize((224, 224))
			if images.shape[2] == 299:
				images = transform(images)

		outputs = model(images)
		if deit:
			outputs = outputs[0]

		_, pre = torch.max(outputs.data, 1)

		total += 1
		correct += (pre == labels).sum()

	acc = (100 * float(correct) / (total * 10))
	sr = 100 - acc
	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total * 10)))
	print('success rate: ' + str(sr))
	return sr


if __name__ == '__main__':
	models = utils.get_target_models()
	model_names = utils.get_target_model_names()

	s_rates = []
	for i in range(len(models)):
		model = models[i]
		model_name = model_names[i]

		print()
		print('Evaluating for ' + model_name)

		if model_name == 'deit-s' or model_name == 'deit-t' or model_name == 'deit-b':
			deit = True
		else:
			deit = False

		if model_name == 'deit-s' or model_name == 'deit-t' or model_name == 'deit-b' or model_name == 'vit-t' or model_name == 'vit-b':
			is_transform = True
		else:
			is_transform = False

		s_rate = evaluate(model, deit, is_transform)
		s_rates.append(s_rate)
		del model
		torch.cuda.empty_cache()

	print(s_rates)