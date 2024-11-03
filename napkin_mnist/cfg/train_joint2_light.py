""" Builds a (pytorch) dataset with data constructed according to the structural equation model

U1  := Unif({0,1,2,3,4,5}) where this is 0->Red, 1->Green, 2->Blue, 3->Yellow, 4->Magenta, 5->Cyan
U2  := Unif({0,1,2}) where 0->Thin, 1->Regular, 2-> Thick
D   := The digit [0..9]
W1  := an MNIST image with (digit:D, Color:U1, Thickness:U2)
W2a := a DIGIT, W1.digit
W2b := a MODIFIED COLOR CODE, taken from U1 // 3
X   := An MNIST image, rotated 90 degrees with (digit:W2a, color: W2b, thickness:U2)
Y   := An MNIST image, reflected, with (digit: X.digit, color: U1, thickness:X.thickness)


So the way we will form this is:
1.  Sample U1, U2, D
2.  Sample W1: pick random mnist image from D, apply color:U1, thickness:U2
3a. Sample W2a with support 10: take digit D and apply massart noise to it
3b. Sample W2b with support 2: take color U1, and transform to U1 // 3 (and apply massart noise)
4.  Sample X: pick random mnist image with Digit W2a, apply color W2b, thickness U2 (randomly massart noise this, too). Rotate 90 degrees
5.  Sample Y: take X and recolor with U2. Reflect. Apply random Massart, too


(Note: we'll do all the data preparation in numpy because we gotta use morphoMNIST to do these things)
"""


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
from embedding import ConditionalEmbedding, MNISTEmbedding, JointEmbedding2
from Scheduler import GradualWarmupScheduler

import sys; sys.path.append('../retrain_trick'); sys.path.append('../Morpho-MNIST')
#from dataloader_cifar import load_data, transback
from dataloader_pickle import transback, PickleDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
import torch.nn.functional as F
from torch import Tensor
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler





parser = argparse.ArgumentParser(description='test for diffusion model')
parser.add_argument('--exp_name', type=str, default="testP(Y,X|W1,W2)")
parser.add_argument('--train_pkl', type=str, default="../napkin_mnist/base_data/napkin_mnist_train.pkl")
parser.add_argument('--val_pkl', type=str, default="../napkin_mnist/base_data/napkin_mnist_val.pkl")
parser.add_argument('--interval',type=int,default=50,help='epoch interval between two evaluations')
parser.add_argument('--moddir',type=str,default='../napkin_mnist/synthetic_model',help='model addresses')
parser.add_argument('--samdir',type=str,default='../napkin_mnist/synthetic_model_evals',help='sample addresses')
parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
parser.add_argument('--epoch',type=int,default=1001,help='epochs for training')

parser.add_argument('--datakey',  type=str, default="X,Y", help='comma separated list of keys to be concatenated in channel dimension to make a single joint model')
parser.add_argument('--labkey_0', type=str, default="W2a", help='which key contains the discrete conditioning')
parser.add_argument('--labkey_1', type=str, default="W2b", help='which key contains the discrete conditioning')
parser.add_argument('--condkey', type=str, default="W1", help='which key contains the image conditioning')
parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
parser.add_argument('--inch',type=int,default=6,help='input channels for Unet model')
parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
parser.add_argument('--outch',type=int,default=6,help='output channels for Unet model')
parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
parser.add_argument('--cdim',type=int,default=64,help='dimension of conditional embedding')
parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
parser.add_argument('--dtype',default=torch.float32)
parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
parser.add_argument('--clsnum_0',type=int,default=10,help='num of label classes')
parser.add_argument('--clsnum_1',type=int,default=2,help='num of label classes')
parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')


# args = parser.parse_args()
params, unknown = parser.parse_known_args()

print('exp name', params.exp_name)

params.moddir= params.moddir+'/'+params.exp_name
params.samdir= params.samdir+'/'+params.exp_name


def cycler(loader):
    while True:
        for batch in loader:
            yield batch


def load_data(dataset: PickleDataset, batchsize: int)-> tuple[DataLoader, DistributedSampler]:
        trainloader = DataLoader(dataset,
                                 batch_size=batchsize,
                                 shuffle=True,
                                 drop_last=True)
        return trainloader

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
train_data = PickleDataset(params.train_pkl)
val_data = PickleDataset(params.val_pkl)
dataloader = load_data(train_data, params.batchsize)
val_loader = load_data(val_data, params.genbatch // torch.cuda.device_count())
val_cycler = cycler(val_loader)

data_keys = params.datakey.split(',')
datacat = lambda batch: torch.cat([batch[k] for k in data_keys], dim=1).contiguous()
datashape = None

# initialize models
net = Unet(
            in_ch = params.inch,
            mod_ch = params.modch,
            out_ch = params.outch,
            ch_mul = params.chmul,
            num_res_blocks = params.numres,
            cdim = params.cdim,
            use_conv = params.useconv,
            droprate = params.droprate,
            dtype = params.dtype
        )



cemblayer = JointEmbedding2(num_labels_0=params.clsnum_0, num_labels_1=params.clsnum_1,   #matt
                           d_model=params.cdim, channels=3,
                           dim=params.cdim, hw=32).to(device)

lastpath = os.path.join(params.moddir,'last_epoch.pt')
if os.path.exists(lastpath):
    lastepc = torch.load(lastpath)['last_epoch']
    # load checkpoints
    checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    cemblayer.load_state_dict(checkpoint['cemblayer'])
else:
    lastepc = 0

betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
diffusion = GaussianDiffusion(
                dtype = params.dtype,
                model = net,
                betas = betas,
                w = params.w,
                v = params.v,
                device = device
            )



# optimizer settings
optimizer = torch.optim.AdamW(
                itertools.chain(
                    diffusion.model.parameters(),
                    cemblayer.parameters()
                ),
                lr = params.lr,
                weight_decay = 1e-4
            )

cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer = optimizer,
                        T_max = params.epoch,
                        eta_min = 0,
                        last_epoch = -1
                    )
warmUpScheduler = GradualWarmupScheduler(
                        optimizer = optimizer,
                        multiplier = params.multiplier,
                        warm_epoch = params.epoch // 10,
                        after_scheduler = cosineScheduler,
                        last_epoch = lastepc
                    )
if lastepc != 0:
    optimizer.load_state_dict(checkpoint['optimizer'])
    warmUpScheduler.load_state_dict(checkpoint['scheduler'])


# training
cnt = torch.cuda.device_count()
for epc in range(lastepc, params.epoch):
    # turn into train mode
    diffusion.model.train()
    cemblayer.train()
    # batch iterations
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for batch in tqdmDataLoader:
            optimizer.zero_grad()
            x_0 = datacat(batch).to(device)
            datashape = tuple(x_0.shape[1:])
            b = x_0.shape[0]
            cemb = cemblayer(batch[params.condkey].to(device),  #matt
                             batch[params.labkey_0].to(device),
                             batch[params.labkey_1].to(device))
            cemb = F.dropout1d(cemb, params.threshold)

            loss = diffusion.trainloss(x_0, cemb = cemb)
            loss.backward()
            optimizer.step()
            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "loss: ": loss.item(),
                    "batch per device: ":x_0.shape[0],
                    "img shape: ": x_0.shape[1:],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }
            )


    warmUpScheduler.step()
    torch.cuda.empty_cache()

    if (epc + 1) % params.interval == 0:
        os.makedirs(params.moddir, exist_ok=True)
        os.makedirs(params.samdir, exist_ok=True)

        diffusion.model.eval()
        cemblayer.eval()
        # generating samples
        # Generates genbatch pictures in 2 columns
        # column 0: conditioning image
        # column 1: generated image (1)
        # column 2: generated image (2)


        all_conds = []
        all_samples = []
        each_device_batch = params.genbatch // cnt
        val_batch = next(val_cycler)
        with torch.no_grad():
            cond = val_batch[params.condkey].to(device)
            cemb = cemblayer(cond,                          #matt
                             val_batch[params.labkey_0].to(device),
                             val_batch[params.labkey_1].to(device))
            genshape = (each_device_batch ,) + datashape
            if params.ddim:
                generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
            else:
                generated = diffusion.sample(genshape, cemb = cemb)

            cond = transback(cond)
            img = transback(generated)


            final_imgs = torch.cat([cond, img], dim=1) #(b, 9, 32, 32)   #matt
            final_imgs = final_imgs.reshape(-1, 3, 32, 32).contiguous()
            save_image(final_imgs, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = 3)
            print('Image saved as ',os.path.join(params.samdir, f'generated_{epc+1}_pict.png'))


        # save checkpoints
        checkpoint = {
                    'net':diffusion.model.state_dict(),
                    'cemblayer':cemblayer.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':warmUpScheduler.state_dict()
                }
        torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
        torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        print('Model saved as ',os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))


    torch.cuda.empty_cache()
