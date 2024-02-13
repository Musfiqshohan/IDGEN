import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding, MNISTEmbedding, JointEmbedding
from Scheduler import GradualWarmupScheduler

#from dataloader_cifar import load_data, transback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
import torch.nn.functional as F 

if __name__ == '__main__':
	while True:
		continue