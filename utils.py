import torch
import torchshow


def combine_pts():
	data_clt = torch.tensor([])
	for i in range(500):
		print(i)
		data = torch.load('chunk/tensor' + str(i) + '.pt').cpu()
		data_clt = torch.cat((data_clt, data), 0)

	print(data_clt.shape)
	torch.save(data_clt, 'test_data.pt')


combine_pts()