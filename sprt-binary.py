import numpy as np
from itertools import cycle
import torch

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import dataloader as dt
import torchshow as ts
import timm
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

np.set_printoptions(suppress=True)


def eliminate_zeros(arr):
	for h in range(len(arr)):
		if arr[h] == float(0):
			arr[h] = float(0.0000001)
	return arr


batch_size = 10
upper_bound = 5
lower_bound = -5


# load first adv dataset
#loader_fgsm = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-1/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-1/test_labels.pt')
loader_cw = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt')
#loader_pgd = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt')

# load second adv dataset
loader_patchfool = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-s/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-s/test_labels.pt')


# define surrogate model
device = torch.device("cuda")

model = models.resnet50(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
#model = models.inception_v3(pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)
deit = False
is_transform = False

model.eval()


if __name__ == '__main__':
	s = 0
	summ = 0
	results = []

	# get adv dataset's accuracy and combine them
	for i, ((X1, y1), (X2, y2)) in enumerate(zip(loader_cw, cycle(loader_patchfool))):
		print(str(i) + " of " + str(len(loader_patchfool)))

		if i == 245:
			print('here')

		X1 = X1.to(device)
		y1 = y1.to(device)
		X2 = X2.to(device)
		y2 = y2.to(device)

		if is_transform:
			transform = transforms.Resize((224, 224))
			X1 = transform(X1)

		model.eval()
		with torch.no_grad():
			out = model(X1)

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
			delta = 1 - prob
			deltas1.append(delta)

		deltas1 = np.asarray(deltas1)
		#print(deltas1.shape)

		model.eval()
		with torch.no_grad():
			out = model(X2)

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
			delta = 1 - prob
			deltas2.append(delta)

		deltas2 = np.asarray(deltas2)
		#print(deltas2.shape)

		deltas1 = eliminate_zeros(deltas1)
		deltas2 = eliminate_zeros(deltas2)

		deltas = np.log(deltas1 / deltas2)

		for k in range(deltas.shape[0]):
			if deltas[k] < 0:
				s = s - 1
			else:
				s = s + 1

		summ = deltas.sum() + summ
		results.append(summ)

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

	#results = np.asarray(results)
	np.save('result', results)