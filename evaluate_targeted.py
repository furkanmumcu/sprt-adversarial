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
	#model = models.resnet50(pretrained=True).to(device)
	model = models.resnet101(pretrained=True).to(device)
	#model = models.vgg16(pretrained=True).to(device)
	#model = models.inception_v3(pretrained=True).to(device)
	#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
	#model = deit_small_patch16_224(pretrained=True).to(device)
	#model = deit_tiny_patch16_224(pretrained=True).to(device)
	#model = deit_base_patch16_224(pretrained=True).to(device)
	deit = False
	is_transform = False

	#loader_pgd_inc_t = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-targeted/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-targeted/inception/test_labels.pt')
	#loader_pgd_inc_t_labels = torch.load('data/test_data_1/sprt-test-set-pgd-targeted/inception/test_tlabels.pt')

	loader_pgd_resnet_t = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-targeted/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-targeted/resnet/test_labels.pt')
	loader_pgd_resnet_t_labels = torch.load('data/test_data_1/sprt-test-set-pgd-targeted/resnet/test_tlabels.pt')

	print("True Image & Predicted Label")

	model.eval()

	correct = 0
	total = 0
	target_label_success = 0

	for i, (images, labels) in enumerate(loader_pgd_resnet_t):
		print(str(i) + " of " + str(len(loader_pgd_resnet_t)))

		images = images.to(device)
		labels = labels.to(device)

		tlabels = []
		for j in range(10):  # batch size = 10
			tlabels.append(loader_pgd_resnet_t_labels[(i*10) + j])
		tlabels = np.asarray(tlabels)
		tlabels = torch.from_numpy(tlabels)
		tlabels = tlabels.to(device)
		tlabels = tlabels.type(torch.long)

		if is_transform:
			transform = transforms.Resize((224, 224))
			if images.shape[2] == 299:
				images = transform(images)

		outputs = model(images)
		if deit:
			outputs = outputs[0]

		_, pre = torch.max(outputs.data, 1)

		total += 1
		target_label_success += (pre == tlabels).sum()
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))
	print('target label success: %f %%' % (100 * float(target_label_success) / (total * 10)))