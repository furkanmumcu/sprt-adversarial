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
import utils

np.set_printoptions(suppress=True)

batch_size = 10
upper_bound = 5
lower_bound = -5

# loader_pgd = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt')

# load first adv dataset
#loader_fgsm = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-1/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-1/test_labels.pt')
#loader_pgd = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt', batch=1, shuffle=True)

# load second adv dataset
loader_cw = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt', batch=1, shuffle=True)

# load third adv dataset
#loader_patchfool = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-s/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-s/test_labels.pt', batch=1, shuffle=True)
loader_pna_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna-deits/test_data.pt', 'data/test_data_1/sprt-test-set-pna-deits/test_labels.pt', batch=1, shuffle=True)

loader_pgd_vit = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt', batch=1, shuffle=True)


# define surrogate model
device = torch.device("cuda")

#model = models.resnet50(pretrained=True).to(device)
model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
# model = models.inception_v3(pretrained=True).to(device)
# model = deit_small_patch16_224(pretrained=True).to(device)
deit = False
is_transform = True

model.eval()


def attack(a, b):
	summ1 = 0
	summ2 = 0
	summ3 = 0

	scores1 = []
	scores2 = []
	scores3 = []

	qnumber = 0
	model_decision = ''
	for i, ((X1, y1), (X2, y2), (X3, y3)) in enumerate(zip(loader_cw, loader_pna_deits, cycle(loader_pgd_vit))):
		#print(str(i) + " of " + str(len(loader_patchfool)))

		X1 = X1.to(device)
		y1 = y1.to(device)
		X2 = X2.to(device)
		y2 = y2.to(device)
		X3 = X3.to(device)
		y3 = y3.to(device)

		if is_transform:
			transform = transforms.Resize((224, 224))
			if X1.shape[2] == 299:
				X1 = transform(X1)
			if X2.shape[2] == 299:
				X2 = transform(X2)
			if X3.shape[2] == 299:
				X3 = transform(X3)

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
		# print(deltas1.shape)

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
		# print(deltas2.shape)

		model.eval()
		with torch.no_grad():
			out = model(X3)

		if deit:
			out = out[0]

		_, pre = torch.max(out.data, 1)

		probabilities3 = torch.nn.functional.softmax(out, dim=1)
		top5_prob3, top5_catid3 = torch.topk(probabilities3, 1000)

		# calculate delta_p_1
		deltas3 = []
		for j in range(len(y3)):
			true_label = y3[j].item()
			true_label_index = (top5_catid3[j] == true_label).nonzero(as_tuple=False).item()
			prob = top5_prob3[j][true_label_index].item()
			delta = 1 - prob
			deltas3.append(delta)

		deltas3 = np.asarray(deltas3)
		# print(deltas3.shape)

		deltas1 = utils.eliminate_zeros(deltas1)
		deltas2 = utils.eliminate_zeros(deltas2)
		deltas3 = utils.eliminate_zeros(deltas3)

		deltas1_2 = np.log(deltas1 / deltas2)
		deltas1_3 = np.log(deltas1 / deltas3)
		deltas2_3 = np.log(deltas2 / deltas3)

		summ1 = deltas1_2.sum() + summ1
		summ2 = deltas1_3.sum() + summ2
		summ3 = deltas2_3.sum() + summ3

		scores1.append(summ1)
		scores2.append(summ2)
		scores3.append(summ3)

		qnumber = (i + 1) * 3  # since we are sending 2 images to the target model

		print('# of queries: ' + str(i + 1))
		print('current scores: ' + str(summ1) + ' ' + str(summ2) + ' ' + str(summ3))
		if summ1 <= a:
			print('2nd selected')  # 2nd selected
			model_decision = '2nd'
			break
		elif summ1 >= b:
			print('1st selected')  # 1st selected
			model_decision = '1st'
			break
		#

		if summ3 <= a:
			print('3rd selected')  # 3rd selected
			model_decision = '3rd'
			break
		elif summ3 >= b:
			print('2nd selected')  # 2nd selected
			model_decision = '2nd'
			break
		#

		if summ2 <= a:
			print('3rd selected')  # 3rd selected
			model_decision = '3rd'
			break
		elif summ2 >= b:
			print('1st selected')  # 1st selected
			model_decision = '1st'
			break
		#

	scores = [scores1, scores2, scores3]
	scores = np.asarray(scores)

	return qnumber, model_decision, scores


if __name__ == '__main__':
	#0.4
	#0.01
	#0.00004
	# 0.000000001
	alpha = 0.000000001
	beta = 0.000000001

	a = np.log(beta / (1 - alpha))
	b = np.log((1 - beta) / alpha)
	attack_band = np.add(a, b)

	expectation = '3rd'
	test_count = 1000

	match = 0
	mismatch = 0
	total_query = 0

	n_first = 0
	n_second = 0
	n_third = 0
	for i in range(test_count):
		print()
		print('# of TEST: ' + str(i + 1))
		results = attack(a, b)
		total_query = total_query + results[0]

		if '1st' == results[1]:
			n_first = n_first + 1
		elif '2nd' == results[1]:
			n_second = n_second + 1
		elif '3rd' == results[1]:
			n_third = n_third + 1

		if expectation == results[1]:
			match = match + 1
		else:
			mismatch = mismatch + 1

	# avarege query number
	avg_qnumber = total_query / test_count

	# detection accuracy
	detection_acc = match / test_count

	first_acc = n_first / test_count
	second_acc = n_second / test_count
	third_acc = n_third / test_count

	print()
	print('expectation: ' + expectation + ' 1st detection accuracy: ' + str(first_acc) + ' 2nd detection accuracy: ' + str(second_acc) + ' 3rd detection accuracy: ' + str(third_acc))
	print('a: ' + str(a) + ' b: ' + str(b) + ' attack_band: ' + str(attack_band) + ' alpha: ' + str(alpha) + ' beta: ' + str(beta))
	print('avarage query number: ' + str(avg_qnumber) + ' detection accuracy: ' + str(detection_acc))

	print('done!')
