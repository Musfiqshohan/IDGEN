from torch import Tensor
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


import pickle


class PickleDataset(Dataset):
	def __init__(self, pkl_file=None, data_dict=None):
		if data_dict is None:
			self.data = pickle.load(open(pkl_file, 'rb'))
		else:
			self.data = data_dict

	def __len__(self):
		return next(iter(self.data.values())).shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		return {k: v[idx] for k,v in self.data.items()}


def load_data(dataset: PickleDataset, batchsize: int, numworkers:int)-> tuple[DataLoader, DistributedSampler]:
		sampler = DistributedSampler(dataset, shuffle=True)
		trainloader = DataLoader(dataset,
							     batch_size=batchsize,
							     num_workers=numworkers,
							     sampler=sampler,
							     drop_last=True)
		return trainloader, sampler

def transback(data:Tensor) -> Tensor:
	return data/2 + 0.5