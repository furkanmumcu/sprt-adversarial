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

np.set_printoptions(suppress=True)

#0.000000001 20


# load first adv dataset
#loader_fgsm = dt.get_loaders_v2('data/test_data_1/sprt-test-set-fgsm-1/test_data.pt', 'data/test_data_1/sprt-test-set-fgsm-1/test_labels.pt')
#loader_cw = dt.get_loaders_v2('data/test_data_1/sprt-test-set-cw-1/test_data.pt', 'data/test_data_1/sprt-test-set-cw-1/test_labels.pt')
#loader_pgd = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/resnet/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/resnet/test_labels.pt', batch=1)
loader_pgd_inception = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pgd-1/inception/test_data.pt', 'data/test_data_1/sprt-test-set-pgd-1/inception/test_labels.pt', batch=1, shuffle=True)

# load second adv dataset
#loader_patchfool = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pfool-s/test_data.pt', 'data/test_data_1/sprt-test-set-pfool-s/test_labels.pt', batch=1)
loader_pna = dt.get_loaders_v2('data/test_data_1/sprt-test-set-pna/test_data.pt', 'data/test_data_1/sprt-test-set-pna/test_labels.pt', batch=1, shuffle=True)

# define surrogate model
device = torch.device("cuda")

#model = models.resnet50(pretrained=True).to(device)
#model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
#model = models.inception_v3(pretrained=True).to(device)
model = models.vgg16(pretrained=True).to(device)
#model = deit_small_patch16_224(pretrained=True).to(device)
deit = False
is_transform = False

model.eval()


def attack(a, b):
	summ = 0
	scores = []

	qnumber = 0
	model_decision = ''
	# get adv dataset's accuracy and combine them
	for i, ((X1, y1), (X2, y2)) in enumerate(zip(loader_pgd_inception, cycle(loader_pna))):
		#print(str(i) + " of " + str(len(loader_pna)))

		if i == 245:
			print('here')

		X1 = X1.to(device)
		y1 = y1.to(device)
		X2 = X2.to(device)
		y2 = y2.to(device)

		if is_transform:
			transform = transforms.Resize((224, 224))
			if X1.shape[2] == 299:
				X1 = transform(X1)
			if X2.shape[2] == 299:
				X2 = transform(X2)

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

		deltas1 = utils.eliminate_zeros(deltas1)
		deltas2 = utils.eliminate_zeros(deltas2)

		deltas = np.log(deltas1 / deltas2)

		summ = deltas.sum() + summ
		scores.append(summ)

		print('# of query: ' + str(i+1))
		print('current score: ' + str(summ))

		if summ <= a:
			print('this is a Transformer')
			model_decision = 'tr'
			break
		elif summ >= b:
			print('this is a CNN')
			model_decision = 'cnn'
			break
		else:
			print('continue to test')

		qnumber = i + 1

	scores = np.asarray(scores)
	return qnumber, model_decision, scores
	#np.save('result', results)


if __name__ == '__main__':
	alpha = 0.000000001
	beta = 0.000000001

	a = np.log(beta / (1 - alpha))
	b = np.log((1 - beta) / alpha)
	attack_band = np.add(a, b)

	expectation = 'cnn'
	test_count = 1000

	match = 0
	mismatch = 0
	total_query = 0
	for i in range(test_count):
		print()
		print('# of TEST: ' + str(i+1))
		results = attack(a, b)
		total_query = total_query + results[0]
		if expectation == results[1]:
			match = match + 1
		else:
			mismatch = mismatch + 1

	#avarege query number
	avg_qnumber = total_query / test_count

	#detection accuracy
	detection_acc = match / test_count

	print('a: ' + str(a) + ' b: ' + str(b) + ' attack_band: ' + str(attack_band) + ' alpha: ' + str(alpha) + ' beta: ' + str(beta))
	print('avarage query number: ' + str(avg_qnumber) + ' detection accuracy: ' + str(detection_acc))

	print('done!')