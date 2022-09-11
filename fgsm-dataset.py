import torch
import torch.nn as nn
from torchvision import models
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224


import dataloader as dt

test_dict = 'C:/Users/furkan/Desktop/projects/combine-attack/data/test_data_1/test_dict.txt'
dct = dt.get_test_dict(test_dict)


def fgsm_attack(model, loss, images, labels, eps):
	images = images.to(device)
	labels = labels.to(device)
	images.requires_grad = True

	outputs = model(images)
	if deit:
		outputs = outputs[0]

	model.zero_grad()
	cost = loss(outputs, labels).to(device)
	cost.backward()

	attack_images = images + eps * images.grad.sign()
	attack_images = torch.clamp(attack_images, 0, 1)

	return attack_images


if __name__ == '__main__':
	test_data = 'C:/Users/furkan/Desktop/projects/combine-attack/data/test_data_1/sprt-test-set'
	eps = 0.007  # 0.007
	use_cuda = True

	device = torch.device("cuda" if use_cuda else "cpu")
	#model = models.resnet50(pretrained=True).to(device)
	model = deit_small_patch16_224(pretrained=True).to(device)

	deit = True

	dloader = dt.get_loaders(test_data)
	model.eval()

	print("Attack Image & Predicted Label")

	correct = 0
	total = 0
	loss = nn.CrossEntropyLoss()

	#empty tensors for saving adv data and corresponding labels
	#data_clt = torch.tensor([])
	#data_clt = data_clt.to(device)
	#labels_clt = torch.tensor([])
	#labels_clt = labels_clt.to(device)

	for i, (images, labels) in enumerate(dloader):
		print(str(i) + " of " + str(len(dloader)))
		labels = torch.ones(10) * dct[str(int(labels[0]))]
		labels = labels.type(torch.long)

		images = fgsm_attack(model, loss, images, labels, eps).to(device)
		labels = labels.to(device)

		#save adv data and corresponding labels
		#data_clt = torch.cat((data_clt, images.cpu().detach()), 0)
		#labels_clt = torch.cat((labels_clt, labels.cpu().detach()), 0)

		outputs = model(images)
		if deit:
			outputs = outputs[0]

		torch.save(images, 'chunk/tensor' + str(i) + '.pt')

		_, pre = torch.max(outputs.data, 1)
		total += 1
		correct += (pre == labels).sum()

	print('Accuracy of test text: %f %%' % (100 * float(correct) / (total*10)))

	#print(data_clt.shape)
	#print(labels_clt.shape)

	#torch.save(data_clt, 'test_data.pt')
	#torch.save(labels_clt, 'test_labels.pt')

