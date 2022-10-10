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
	#model = models.inception_v3(pretrained=True).to(device)
	#model = models.vgg16(pretrained=True).to(device)
	#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
	#model = deit_small_patch16_224(pretrained=True).to(device)
	#model = deit_tiny_patch16_224(pretrained=True).to(device)
	#model = deit_base_patch16_224(pretrained=True).to(device)

	#loader_adv_fgsm = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-1/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-1/test_labels.pt')
	#loader_clean = dt.get_loaders_v2('data/test_data_1/sprt-test-set-clean-pt-224/test_data.pt', 'data/test_data_1/sprt-test-set-clean-pt-224/test_labels.pt')
	#loader_clean_299 = dt.get_loaders_v2('data/test_data_1/sprt-test-set-clean-pt-299/test_data.pt', 'data/test_data_1/sprt-test-set-clean-pt-299/test_labels.pt')

	#loader_pgd_resnet = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt')
	#loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt')
	#loader_pgd_deit_s = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/deit-s/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/deit-s/test_labels.pt')
	#loader_pgd_vit_base = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt')
	#loader_pna = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna/test_data.pt', 'data/test_data_1/sprt-test-set-pna/test_labels.pt')
	#loader_pna_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna-deits/test_data.pt', 'data/test_data_1/sprt-test-set-pna-deits/test_labels.pt')
	#loader_cw = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt')
	#loader_patchfool_s = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-s/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-s/test_labels.pt')
	#loader_patchfool_t = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-t/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-t/test_labels.pt')
	#loader_patchfool_b = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-b/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-b/test_labels.pt')

	############################################################################################################################################################
	############################################################################################################################################################
	############################################################################################################################################################
	############################################################################################################################################################
	############################################################################################################################################################

	#loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt')
	#loader_pgd_deit_s = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/deit-s/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/deit-s/test_labels.pt')
	#loader_pgd_vit_base = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt')
	#loader_pgd_resnet = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt')
	#loader_pgd_vgg16 = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vgg16/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vgg16/test_labels.pt')

	#loader_fgsm_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-inception/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-inception/test_labels.pt')
	#loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt')
	#loader_cw_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt')
	loader_pna_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna-inception/test_data.pt', 'data/test_data_1/sprt-test-set-pna-inception/test_labels.pt')

	###

	#model = models.inception_v3(pretrained=True).to(device)
	#model = deit_small_patch16_224(pretrained=True).to(device)
	#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
	#model = models.resnet50(pretrained=True).to(device)
	#model = models.vgg16(pretrained=True).to(device)

	#model = deit_tiny_patch16_224(pretrained=True).to(device)
	#model = deit_base_patch16_224(pretrained=True).to(device)
	model = models.resnet152(pretrained=True).to(device)
	#model = models.vgg19(pretrained=True).to(device)
	#model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
	#model = timm.create_model('levit_256', pretrained=True).to(device)

	deit = False
	is_transform = False

	print("True Image & Predicted Label")

	model.eval()

	correct = 0
	total = 0

	for i, (images, labels) in enumerate(loader_pna_inception):
		print(str(i) + " of " + str(len(loader_pna_inception)))

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

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))