import numpy as np
from itertools import cycle
import torch
import utils

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import dataloader as dt
import torchshow as ts
import timm
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

from itertools import combinations


np.set_printoptions(suppress=True)

#0.000000001 20

b_size = 10

# load first adv dataset
#loader_fgsm = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-1/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-1/test_labels.pt')
#loader_cw = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt')
loader_pgd_resnet = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/deit-s/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/deit-s/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_vit = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_vgg16 = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt', batch=b_size, shuffle=True)

# load second adv dataset
#loader_patchfool = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-s/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-s/test_labels.pt', batch=b_size, shuffle=True)
#loader_pna = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna/test_data.pt', 'data/test_data_1/sprt-test-set-pna/test_labels.pt', batch=5, shuffle=True)

# define surrogate model
device = torch.device("cuda")

#model = models.resnet50(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
model = models.inception_v3(pretrained=True).to(device)
#model = models.vgg16(pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)
deit = False
is_transform = False

model.eval()

multi_case = '3'

#strategy_index_3 = [1, 2, 3]
strategy_index_4 = [1, 2, 3, 4]
strategy_index_5 = [1, 2, 3, 4, 5]

#strategy_pairs3 = sorted(map(sorted, combinations(set(strategy_index_3), 2)))
strategy_pairs4 = sorted(map(sorted, combinations(set(strategy_index_4), 2)))
strategy_pairs5 = sorted(map(sorted, combinations(set(strategy_index_5), 2)))

#print(strategy_pairs3)
print(strategy_pairs4)
print(strategy_pairs5)

#strategies = []


def calculate_pair_scores(strategies, strategy_pairs, model, scores):
	for j in range(len(strategy_pairs)):  # 3 for
		X_first = strategies[strategy_pairs[j][0]][0]
		y_first = strategies[strategy_pairs[j][0]][1]

		X_second = strategies[strategy_pairs[j][1]][0]
		y_second = strategies[strategy_pairs[j][1]][1]

		model.eval()
		with torch.no_grad():
			out = model(X_first)

		if deit:
			out = out[0]

		_, pre = torch.max(out.data, 1)

		deltas1 = []
		score1 = 1 - (((pre == y_first).sum()).item() / b_size)
		deltas1.append(score1)
		deltas1 = np.asarray(deltas1).astype(np.float32)

		#
		model.eval()
		with torch.no_grad():
			out = model(X_second)

		if deit:
			out = out[0]

		_, pre = torch.max(out.data, 1)

		deltas2 = []
		score2 = 1 - (((pre == y_second).sum()).item() / b_size)
		deltas2.append(score2)
		deltas2 = np.asarray(deltas2).astype(np.float32)

		#
		deltas1 = utils.eliminate_zeros(deltas1)
		deltas2 = utils.eliminate_zeros(deltas2)
		deltas = np.log(deltas1 / deltas2)

		scores[j] = deltas.sum() + scores[j]
	return scores


def attack(a, b, multi_case):
	if multi_case == '3':
		print('multi_case 3')
		scores = [0, 0, 0]
		selected = 0
		qnumber = 0
		breaked = False
		strategy_index_3 = [0, 1, 2]
		strategy_pairs3 = sorted(map(sorted, combinations(set(strategy_index_3), 2)))
		for i, ((X1, y1), (X2, y2), (X3, y3)) in enumerate(zip(loader_pgd_inception, loader_pgd_deits, cycle(loader_pgd_vit))):
			if breaked:
				break

			print(str(i) + " of " + str(len(loader_pgd_vit)))

			X1 = X1.to(device)
			y1 = y1.to(device)
			X2 = X2.to(device)
			y2 = y2.to(device)
			X3 = X2.to(device)
			y3 = y2.to(device)

			if is_transform:
				transform = transforms.Resize((224, 224))
				if X1.shape[2] == 299:
					X1 = transform(X1)
				if X2.shape[2] == 299:
					X2 = transform(X2)
				if X3.shape[2] == 299:
					X3 = transform(X3)

			strategies = [(X1, y1), (X2, y2), (X3, y3)]

			'''
			for j in range(len(strategy_pairs3)):  # 3 for
				X_first = strategies[strategy_pairs3[j][0]][0]
				y_first = strategies[strategy_pairs3[j][0]][1]

				X_second = strategies[strategy_pairs3[j][1]][0]
				y_second = strategies[strategy_pairs3[j][1]][1]

				model.eval()
				with torch.no_grad():
					out = model(X_first)

				if deit:
					out = out[0]

				_, pre = torch.max(out.data, 1)

				deltas1 = []
				score1 = 1 - (((pre == y_first).sum()).item() / b_size)
				deltas1.append(score1)
				deltas1 = np.asarray(deltas1).astype(np.float32)

				#
				model.eval()
				with torch.no_grad():
					out = model(X_second)

				if deit:
					out = out[0]

				_, pre = torch.max(out.data, 1)

				deltas2 = []
				score2 = 1 - (((pre == y_second).sum()).item() / b_size)
				deltas2.append(score2)
				deltas2 = np.asarray(deltas2).astype(np.float32)

				#
				deltas1 = utils.eliminate_zeros(deltas1)
				deltas2 = utils.eliminate_zeros(deltas2)
				deltas = np.log(deltas1 / deltas2)

				scores[j] = deltas.sum() + scores[j]
			'''

			scores = calculate_pair_scores(strategies, strategy_pairs3, model, scores)

			qnumber = (i + 1) * (3 * b_size)  # since we are sending  images to the target model
			print('current scores: ')
			print(scores)
			print('# of queries: ' + str(i + 1))
			# 1_2
			for j in range(len(scores)):
				if scores[j] <= a:
					selected = strategy_pairs3[j][1] + 1
					print(str(selected) + ' is selected')  # 2nd selected
					breaked = True
					break
				elif scores[j] >= b:
					selected = strategy_pairs3[j][0] + 1
					print(str(selected) + ' is selected')  # 1st selected
					breaked = True
					break
				else:
					print('continue to test')

		return qnumber, selected

	if multi_case == '4':
		print('multi_case 4')
		for i, ((X1, y1), (X2, y2), (X3, y3), (X4, y4)) in enumerate(zip(loader_pgd_inception, loader_pgd_deits, loader_pgd_vit, cycle(loader_pgd_resnet))):
			print(str(i) + " of " + str(len(loader_pgd_resnet)))

			X1 = X1.to(device)
			y1 = y1.to(device)
			X2 = X2.to(device)
			y2 = y2.to(device)
			X3 = X3.to(device)
			y3 = y3.to(device)
			X4 = X4.to(device)
			y4 = y4.to(device)

			model.eval()
			if is_transform:
				transform = transforms.Resize((224, 224))
				if X1.shape[2] == 299:
					X1 = transform(X1)
				if X2.shape[2] == 299:
					X2 = transform(X2)
				if X3.shape[2] == 299:
					X3 = transform(X3)
				if X4.shape[2] == 299:
					X4 = transform(X4)

	if multi_case == '5':
		print('multi_case 3')
		for i, ((X1, y1), (X2, y2), (X3, y3), (X4, y4), (X5, y5)) in enumerate(zip(loader_pgd_inception, loader_pgd_deits, loader_pgd_vit, cycle(loader_pgd_vgg16))):
			print(str(i) + " of " + str(len(loader_pgd_vgg16)))

			X1 = X1.to(device)
			y1 = y1.to(device)
			X2 = X2.to(device)
			y2 = y2.to(device)
			X3 = X3.to(device)
			y3 = y3.to(device)
			X4 = X4.to(device)
			y4 = y4.to(device)
			X5 = X5.to(device)
			y5 = y5.to(device)

			model.eval()
			if is_transform:
				transform = transforms.Resize((224, 224))
				if X1.shape[2] == 299:
					X1 = transform(X1)
				if X2.shape[2] == 299:
					X2 = transform(X2)
				if X3.shape[2] == 299:
					X3 = transform(X3)
				if X4.shape[2] == 299:
					X4 = transform(X4)
				if X5.shape[2] == 299:
					X5 = transform(X5)


if __name__ == '__main__':
	#0.4
	#0.01
	#0.00004
	# 0.000000001

	alpha = 0.00004
	beta = 0.00004

	expectation = 1

	test_count = 1
	multi_case = '3'

	a = np.log(beta / (1 - alpha))
	b = np.log((1 - beta) / alpha)
	attack_band = np.add(a, b)

	total_query = 0

	n_first = 0
	n_second = 0
	n_third = 0
	n_fourth = 0
	n_fifth = 0
	for i in range(test_count):
		print()
		print('# of TEST: ' + str(i+1))

		results = attack(a, b, multi_case)

		total_query = total_query + results[0]

		if 1 == results[1]:
			n_first = n_first + 1
		elif 2 == results[1]:
			n_second = n_second + 1
		elif 3 == results[1]:
			n_third = n_third + 1
		elif 4 == results[1]:
			n_fourth = n_fourth + 1
		elif 5 == results[1]:
			n_fifth = n_fifth + 1

	# avarege query number
	avg_qnumber = total_query / test_count

	first_acc = n_first / test_count
	second_acc = n_second / test_count
	third_acc = n_third / test_count

	print('expectation: ' + str(expectation) + ' 1st detection accuracy: ' + str(first_acc) + ' 2nd detection accuracy: ' + str(second_acc) + ' 3rd detection accuracy: ' + str(third_acc))