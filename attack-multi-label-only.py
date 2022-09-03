import numpy as np
from itertools import cycle
import torch
import utils
import operator
import sys

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import dataloader as dt
import torchshow as ts
import timm
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224

from itertools import combinations
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--target', type=str,
                    help='An optional integer argument')

parser.add_argument('--threshold', type=str,
                    help='An optional integer argument')

parser.add_argument('--multi_case', type=str,
                    help='An optional integer argument')
parser.add_argument('--deit', type=str,
                    help='An optional integer argument')
parser.add_argument('--is_transform', type=str,
                    help='An optional integer argument')
parser.add_argument('--expectation', type=int,
                    help='An optional integer argument', default=1)

args = parser.parse_args()

print(args.target)
print(args.threshold)
print(args.multi_case)
print(args.deit)
print(args.is_transform)


np.set_printoptions(suppress=True)
b_size = 5
device = torch.device("cuda")

# attack strategies

loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_deits = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/deit-s/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/deit-s/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_vit = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_resnet = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt', batch=b_size, shuffle=True)
loader_pgd_vgg16 = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/vit-base/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/vit-base/test_labels.pt', batch=b_size, shuffle=True)


# target model

#model = models.inception_v3(pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
#model = models.resnet50(pretrained=True).to(device)
#model = models.vgg16(pretrained=True).to(device)

#model = deit_tiny_patch16_224(pretrained=True).to(device)
#model = deit_base_patch16_224(pretrained=True).to(device)
#model = models.resnet152(pretrained=True).to(device)
#model = models.vgg19(pretrained=True).to(device)
#model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
#model = timm.create_model('levit_256', pretrained=True).to(device)

model = utils.get_target_models_v2()[args.target]
model.eval()

#deit = False
#is_transform = False
deit = args.deit == 'True'
is_transform = args.is_transform == 'True'


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


def decide_strategy(a, b, scores, strategy_pairs):
	# 1_2
	selected = -1
	for j in range(len(scores)):
		if scores[j] <= a:
			selected = strategy_pairs[j][1] + 1
			print(str(selected) + ' is selected')  # 2nd selected
			break
		elif scores[j] >= b:
			selected = strategy_pairs[j][0] + 1
			print(str(selected) + ' is selected')  # 1st selected
			break
		else:
			selected = -1

	return selected


def decide_strategy_v2(a, b, scores, strategy_pairs):
	# 1_2
	selected = -1
	selected_list = {}
	for j in range(len(scores)):
		if scores[j] <= a:
			selected = strategy_pairs[j][1] + 1
			selected_list[selected] = a - scores[j]
		elif scores[j] >= b:
			selected = strategy_pairs[j][0] + 1
			selected_list[selected] = scores[j] - b
		else:
			if len(selected_list) == 0:
				selected = -1
			elif len(selected_list) == 1:
				selected = next(iter(selected_list))
			else:
				sorted_selected_list = dict(sorted(selected_list.items(), key=operator.itemgetter(1), reverse=True))
				selected = next(iter(sorted_selected_list))

	if selected != -1:
		print(str(selected) + ' is selected')
	return selected


def attack(a, b, multi_case):
	if multi_case == '3':
		print('multi_case 3')
		scores = [0, 0, 0]
		selected = 0
		qnumber = 0
		strategy_index_3 = [0, 1, 2]
		strategy_pairs3 = sorted(map(sorted, combinations(set(strategy_index_3), 2)))
		for i, ((X1, y1), (X2, y2), (X3, y3)) in enumerate(zip(loader_pgd_inception, loader_pgd_deits, cycle(loader_pgd_vit))):
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

			strategies = [(X1, y1), (X2, y2), (X3, y3)]

			scores = calculate_pair_scores(strategies, strategy_pairs3, model, scores)

			qnumber = (i + 1) * (3 * b_size)  # since we are sending  images to the target model
			print('current scores: ')
			print(scores)
			print('# of queries: ' + str(i + 1))

			selected = decide_strategy_v2(a, b, scores, strategy_pairs3)
			if selected != -1:
				break
			else:
				print('continue to test')

		return qnumber, selected

	if multi_case == '4':
		print('multi_case 4')
		scores = [0, 0, 0, 0, 0, 0]
		selected = 0
		qnumber = 0
		strategy_index_4 = [0, 1, 2, 3]
		strategy_pairs4 = sorted(map(sorted, combinations(set(strategy_index_4), 2)))
		for i, ((X1, y1), (X2, y2), (X3, y3), (X4, y4)) in enumerate(zip(loader_pgd_inception, loader_pgd_deits, loader_pgd_vit, cycle(loader_pgd_resnet))):
			#print(str(i) + " of " + str(len(loader_pgd_resnet)))

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

			strategies = [(X1, y1), (X2, y2), (X3, y3), (X4, y4)]

			scores = calculate_pair_scores(strategies, strategy_pairs4, model, scores)

			qnumber = (i + 1) * (4 * b_size)  # since we are sending  images to the target model
			print('current scores: ')
			print(scores)
			print('# of queries: ' + str(i + 1))

			selected = decide_strategy_v2(a, b, scores, strategy_pairs4)
			if selected != -1:
				break
			else:
				print('continue to test')

		return qnumber, selected

	if multi_case == '5':
		print('multi_case 5')
		scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		selected = 0
		qnumber = 0
		strategy_index_5 = [0, 1, 2, 3, 4]
		strategy_pairs5 = sorted(map(sorted, combinations(set(strategy_index_5), 2)))
		for i, ((X1, y1), (X2, y2), (X3, y3), (X4, y4), (X5, y5)) in enumerate(zip(loader_pgd_inception, loader_pgd_deits, loader_pgd_vit, loader_pgd_resnet, cycle(loader_pgd_vgg16))):
			#print(str(i) + " of " + str(len(loader_pgd_vgg16)))

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

			strategies = [(X1, y1), (X2, y2), (X3, y3), (X4, y4), (X5, y5)]

			scores = calculate_pair_scores(strategies, strategy_pairs5, model, scores)

			qnumber = (i + 1) * (5 * b_size)  # since we are sending  images to the target model
			print('current scores: ')
			print(scores)
			print('# of queries: ' + str(i + 1))

			selected = decide_strategy_v2(a, b, scores, strategy_pairs5)
			if selected != -1:
				break
			else:
				print('continue to test')

		return qnumber, selected


if __name__ == '__main__':
	#0.4
	#0.01
	#0.00004
	#0.000000001

	if args.threshold == 's':
		alpha = 0.4
		beta = 0.4
	elif args.threshold == 'm':
		alpha = 0.01
		beta = 0.01
	elif args.threshold == 'l':
		alpha = 0.00004
		beta = 0.00004
	elif args.threshold == 'xl':
		alpha = 0.000000001
		beta = 0.000000001
	else:
		raise Exception('threshold did not defined')

	expectation = args.expectation

	test_count = 100
	#multi_case = '5'
	multi_case = args.multi_case

	a = np.log(beta / (1 - alpha))
	b = np.log((1 - beta) / alpha)
	attack_band = np.add(a, b)

	total_query = 0
	match = 0
	mismatch = 0

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

		if expectation == results[1]:
			match = match + 1
		else:
			mismatch = mismatch + 1

	# avarege query number
	avg_qnumber = total_query / test_count

	# detection accuracy
	detection_acc = match / test_count

	first_rate = n_first / test_count
	second_rate = n_second / test_count
	third_rate = n_third / test_count
	fourth_rate = n_fourth / test_count
	fifth_rate = n_fifth / test_count

	rates = [first_rate, second_rate, third_rate, fourth_rate, fifth_rate]
	print(rates)

	target_model = 'resnet50'
	strategy_accuries = utils.get_strategy_accuracies()
	success_rate = 0
	for i in range(len(rates)):
		success_rate = success_rate + (rates[i] * (100 - strategy_accuries[i][target_model]))

	print('expectation: ' + str(expectation) + ' 1st detection rate: ' + str(first_rate) + ' 2nd detection rate: ' + str(second_rate) + ' 3rd detection rate: ' + str(third_rate) + ' 4rd detection rate: ' + str(fourth_rate) + ' 5th detection rate: ' + str(fifth_rate))
	print('a: ' + str(a) + ' b: ' + str(b) + ' attack_band: ' + str(attack_band) + ' alpha: ' + str(alpha) + ' beta: ' + str(beta))
	print('avarage query number: ' + str(avg_qnumber) + ' detection accuracy: ' + str(detection_acc))
	print('success rate: ' + str(success_rate))

	print(str(detection_acc * 100) + ' - ' + str(success_rate) + ' - ' + str(avg_qnumber))
	print('na ' + ' - ' + str(success_rate) + ' - ' + str(avg_qnumber))

	line0 = ''.join(str(x) for x in sys.argv[1:])
	line1 = 'expectation: ' + str(expectation) + ' 1st detection rate: ' + str(first_rate) + ' 2nd detection rate: ' + str(second_rate) + ' 3rd detection rate: ' + str(third_rate) + ' 4rd detection rate: ' + str(fourth_rate) + ' 5th detection rate: ' + str(fifth_rate)
	line2 = 'a: ' + str(a) + ' b: ' + str(b) + ' attack_band: ' + str(attack_band) + ' alpha: ' + str(alpha) + ' beta: ' + str(beta)
	line3 = 'avarage query number: ' + str(avg_qnumber) + ' detection accuracy: ' + str(detection_acc)
	line4 = 'success rate: ' + str(success_rate)
	line5 = str(detection_acc * 100) + ' - ' + str(success_rate) + ' - ' + str(avg_qnumber)
	line6 = 'na ' + ' - ' + str(success_rate) + ' - ' + str(avg_qnumber)

	fname = args.multi_case + '-' + args.target + '-' + args.threshold
	lines = [line0, line1, line2, line3, line4, line5, line6]

	with open('outs/' + fname + '.txt', 'w') as f:
		for line in lines:
			f.write(line)
			f.write('\n')