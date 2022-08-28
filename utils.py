import torch
import torchshow


def eliminate_zeros(arr):
	for h in range(len(arr)):
		if arr[h] == float(0):
			arr[h] = float(0.0000001)
	return arr


def get_strategy_accuracies():
	dict_pgd_inception = dict(
		[('inception', 0.02), ('deit-s', 58.5), ('vit-b', 71.5), ('resnet50', 18.88), ('vgg16', 10.42),
		 ('deit-t', 44.72), ('deit-b', 62.42), ('resnet152', 12.88), ('vgg19', 58.5), ('vit-t', 38.56)])
	dict_pgd_deits = dict(
		[('inception', 0.02), ('deit-s', 0), ('vit-b', 43.06), ('resnet50', 18.32), ('vgg16', 9.24), ('deit-t', 6.52),
		 ('deit-b', 13.3), ('resnet152', 28.36), ('vgg19', 10.38), ('vit-t', 9.9)])
	dict_pgd_vit = dict(
		[('inception', 34.26), ('deit-s', 36.92), ('vit-b', 0), ('resnet50', 22), ('vgg16', 11.7), ('deit-t', 31.92),
		 ('deit-b', 34.2), ('resnet152', 31.66), ('vgg19', 13.1), ('vit-t', 22.56)])
	dict_pgd_resnet = dict(
		[('inception', 33.6), ('deit-s', 56.9), ('vit-b', 69.68), ('resnet50', 0), ('vgg16', 7.62), ('deit-t', 42.44),
		 ('deit-b', 61.7), ('resnet152', 13.92), ('vgg19', 8.44), ('vit-t', 34.2)])
	dict_pgd_vgg16 = dict(
		[('inception', 34.64), ('deit-s', 55.34), ('vit-b', 67.3), ('resnet50', 10.58), ('vgg16', 0), ('deit-t', 40.5),
		 ('deit-b', 59.28), ('resnet152', 21.38), ('vgg19', 0.52), ('vit-t', 31.36)])

	return [dict_pgd_inception, dict_pgd_deits, dict_pgd_vit, dict_pgd_resnet, dict_pgd_vgg16]


def combine_pts():
	data_clt = torch.tensor([])
	for i in range(500):
		print(i)
		data = torch.load('chunk/tensor' + str(i) + '.pt').cpu()
		data_clt = torch.cat((data_clt, data), 0)

	print(data_clt.shape)
	torch.save(data_clt, 'test_data.pt')


def combine_pts_targeted():
	data_clt = torch.tensor([])
	for i in range(500):
		print(i)
		data = torch.load('chunk/tensor' + str(i) + '.pt').cpu()
		data_clt = torch.cat((data_clt, data), 0)

	tlabel_clt = torch.tensor([])
	for i in range(500):
		print(i)
		data = torch.load('chunk_tlabel/tensor' + str(i) + '.pt').cpu()
		tlabel_clt = torch.cat((tlabel_clt, data), 0)

	print(data_clt.shape)
	torch.save(data_clt, 'test_data.pt')

	print(tlabel_clt.shape)
	torch.save(tlabel_clt, 'test_tlabels.pt')


#combine_pts()