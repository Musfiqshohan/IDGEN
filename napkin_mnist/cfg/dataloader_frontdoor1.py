from torch import Tensor
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import gzip
import struct
import numpy as np
import pickle


#[mj] -- this whole file is new, modified from dataloader_cifar



def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)


class FrontDoorDataset(Dataset):
    def __init__(self, imgs, A, U, D, transform=None):
        self.imgs = torch.Tensor(load_idx(imgs)) /127.5 - 1.0 # normalize to [-1, 1]
        self.imgs = self.imgs.permute(0,3,1,2).clone() # put in CHW format
        self.A = torch.Tensor(pickle.load(open(A, 'rb'))).view(-1).long()
        self.U = torch.Tensor(pickle.load(open(U, 'rb'))).view(-1).long()
        self.D = torch.Tensor(pickle.load(open(D, 'rb'))).view(-1).long()
        self.transform = transform if transform is not None else lambda x: x
        
    def __len__(self):
        return self.D.numel()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'image': self.transform(self.imgs[idx]),
                  'A': self.A[idx],
                  'U': self.U[idx],
                  'D': self.D[idx]}
        return sample

    def to(self, device):
    	self.imgs = self.imgs.to(device)
    	self.A = self.A.to(device)
    	self.U = self.U.to(device)
    	self.D = self.D.to(device)
    	return self


         


def load_data(batchsize: int, numworkers: int) -> tuple[DataLoader, DistributedSampler]:
	data_train = FrontDoorDataset(imgs='data/DigitImages.gz', A='data/A.pkl', U='data/U.pkl',
								  D='data/D.pkl', transform=None)
	sampler = DistributedSampler(data_train, shuffle=True)
	trainloader = DataLoader(data_train,
		 					 batch_size=batchsize,
		 					 num_workers=numworkers,
		 					 sampler=sampler,
		 					 drop_last=True)
	return trainloader, sampler

def transback(data:Tensor) -> Tensor:
	return data/2 + 0.5