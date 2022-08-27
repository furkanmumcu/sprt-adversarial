import torch
import torchshow


def eliminate_zeros(arr):
	for h in range(len(arr)):
		if arr[h] == float(0):
			arr[h] = float(0.0000001)
	return arr


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