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

import sys; sys.path.append('../Morpho-MNIST')
#from dataloader_cifar import load_data, transback
from dataloader_pickle import load_data, transback, PickleDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
import torch.nn.functional as F 
def cycler(loader):
    while True:
        for batch in loader:
            yield batch


def train(params:argparse.Namespace):
    assert params.genbatch % (torch.cuda.device_count()) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    train_data = PickleDataset(params.train_pkl)
    val_data = PickleDataset(params.val_pkl)
    dataloader, sampler = load_data(train_data, params.batchsize, params.numworkers)
    val_loader, val_sampler = load_data(val_data, params.genbatch // torch.cuda.device_count(), 
                                        params.numworkers)
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


    #cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(device)
    #cemblayer = MNISTEmbedding(channels=3, dim=params.cdim, hw=32).to(device)
    cemblayer = JointEmbedding2(num_labels_0=params.clsnum_0, num_labels_1=params.clsnum_1, 
                                d_model=params.cdim, channels=3,
                                dim=params.cdim, hw=32).to(device)

    #cemblayer = MNISTEmbedding(3, params.cdim, hw=32).to(device)
    # load last epoch
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
    
    # DDP settings 
    diffusion.model = DDP(
                            diffusion.model,
                            device_ids = [local_rank],
                            output_device = local_rank
                        )
    cemblayer = DDP(
                    cemblayer,
                    device_ids = [local_rank],
                    output_device = local_rank
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
        sampler.set_epoch(epc)
        # batch iterations
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = datacat(batch).to(device)
                datashape = tuple(x_0.shape[1:])
                b = x_0.shape[0]
                cemb = cemblayer(batch[params.condkey].to(device),
                                 batch[params.labkey_0].to(device),
                                 batch[params.labkey_1].to(device),
                                 drop_label=params.drop_lab,
                                 drop_image=params.drop_imgcond,
                                 threshold=params.threshold)


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
        # evaluation and save checkpoint
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
                cemb = cemblayer(cond,
                                 val_batch[params.labkey_0].to(device),
                                 val_batch[params.labkey_1].to(device))
                cemb_uncond = cemblayer(cond, 
                                        val_batch[params.labkey_0].to(device),
                                        val_batch[params.labkey_1].to(device),
                                        drop_label=params.drop_lab,
                                        drop_image=params.drop_imgcond,
                                        threshold=999.0)
                genshape = (each_device_batch ,) + datashape
                if params.ddim:
                    generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb, cemb_uncond=cemb_uncond)
                else:
                    generated = diffusion.sample(genshape, cemb = cemb, cemb_uncond=cemb_uncond)

                cond = transback(cond)
                img = transback(generated)

                gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
                all_gather(gathered_samples, img)
                gathered_conds = [torch.zeros_like(cond) for _ in range(get_world_size())]
                all_gather(gathered_conds, cond)

                all_samples.extend([_ for _ in gathered_samples])
                final_samples = torch.cat(all_samples, dim=0)
                all_conds.extend([_ for _ in gathered_conds])
                final_conds = torch.cat(all_conds, dim=0)

                final_imgs = torch.cat([final_conds, final_samples], dim=1) #(b, 9, 32, 32)
                final_imgs = final_imgs.reshape(-1, 3, 32, 32).contiguous()
                #final_imgs = torch.stack([final_conds, final_samples]).permute(1,0,2,3,4).reshape(-1, 3, 32, 32).contiguous()
                if local_rank == 0:
                    #print(final_imgs.shape)
                    save_image(final_imgs, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = 2)
            # save checkpoints
            checkpoint = {
                                'net':diffusion.model.module.state_dict(),
                                'cemblayer':cemblayer.module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()
    destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')
    parser.add_argument('--train_pkl', type=str, required=True)
    parser.add_argument('--val_pkl', type=str, required=True)
    parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
    parser.add_argument('--epoch',type=int,default=1000,help='epochs for training')
    parser.add_argument('--interval',type=int,default=100,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='joint_model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='joint_sample',help='sample addresses')

    parser.add_argument('--datakey',  type=str, required=True, help='comma separated list of keys to be concatenated in channel dimension to make a single joint model')
    parser.add_argument('--labkey_0', type=str, required=True, help='which key contains the discrete conditioning')
    parser.add_argument('--labkey_1', type=str, required=True, help='second key for discrete conditioning')
    parser.add_argument('--condkey', type=str, required=True, help='which key contains the image conditioning')
    parser.add_argument('--drop-lab', type=int, default=1, help='do we sometimes drop the label during cfg training?')
    parser.add_argument('--drop-imgcond', type=int, default=1, help='do we sometimes drop the img conditioner during cfg training?')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
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
    parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
    parser.add_argument('--clsnum_0',type=int,default=10,help='num of label classes')
    parser.add_argument('--clsnum_1', type=int, default=2)
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')

    args = parser.parse_args()

    epoch = [int(s) for s in args.train_pkl.replace('.', '_').split('_') if s.isdigit()][0]
    args.moddir= f'{args.moddir}/wEpoch{epoch}'
    args.samdir= f'{args.samdir}/wEpoch{epoch}'

    train(args)

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu cfg/train_semijoint2.py \
# 	--train_pkl napkin_mnist/synthetic_training_data/synthetic_W1W2XY_1000.pkl\
# 	--val_pkl napkin_mnist/synthetic_training_data/synthetic_W1W2XY_1000.pkl\
# 	--inch 3 \
# 	--outch 3 \
# 	--datakey Y \
# 	--condkey X \
# 	--labkey_0 W2a \
# 	--labkey_1 W2b \
# 	--drop-lab 0 \
# 	--drop-imgcond 1\
# 	--moddir napkin_mnist/final_model_NODROP \
# 	--samdir napkin_mnist/final_model_evals_NODROP \
# 	--epoch 1001 \
# 	--interval 50