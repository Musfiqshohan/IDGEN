import os

from constantFunctions import get_prediction
from tool.utils import available_devices,format_devices
#set device
device = available_devices(threshold=10000,n_devices=4)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
from tool.reproducibility import set_seed
from tool.utils import dict2namespace
import yaml
import torch
from runners.egsde import EGSDE
from tool.interact import set_logger
from models.ddpm import Model
import os
import logging
import numpy as np
import torch
import torch.utils.data as data
from models.ddpm import Model
from datasets import get_dataset,rescale,inverse_rescale
import torchvision.utils as tvu
from functions.denoising import egsde_sample
from guided_diffusion.script_util import create_model,create_dse
from functions.resizer import Resizer
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Classifier.pre_trained import get_classifier

import argparse



if __name__ == '__main__':

    sample_dir= "/local/scratch/a/rahman89/PycharmProjects/STAR_GAN/IDDAG/egsde_celeba_256"
    argsall = argparse.Namespace(
        # testdata_path='/local/scratch/a/rahman89/PycharmProjects/EGSDE/data/celeba_hq/val/male',
        testdata_path='/local/scratch/a/rahman89/PycharmProjects/STAR_GAN/IDDAG/original_celeba_256',
        ckpt='/local/scratch/a/rahman89/PycharmProjects/EGSDE/pretrained_model/celebahq_female_ddpm.pth',
        dsepath='/local/scratch/a/rahman89/PycharmProjects/EGSDE/pretrained_model/male2female_dse.pt',
        config_path='/local/scratch/a/rahman89/PycharmProjects/EGSDE/profiles/male2female/male2female.yml',
        t=500,
        ls=500.0,
        li=2.0,
        s1='cosine',
        s2='neg_l2',
        phase='test',
        root='runs/',
        sample_step=1,
        batch_size=20,
        diffusionmodel='DDPM',
        down_N=32,
        seed=1234)

    task = 'male2female'
    # from profiles.male2female.args import argsall

    # args
    args = argsall
    set_seed(args.seed)
    args.samplepath = os.path.join('runs', task)
    os.makedirs(args.samplepath, exist_ok=True)
    set_logger(args.samplepath, 'sample.txt')




    # config
    with open(args.config_path, "r") as f:
        config_ = yaml.safe_load(f)
    config = dict2namespace(config_)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    egsde = EGSDE(args, config)

    args, config = egsde.args, egsde.config

    if args.diffusionmodel == 'DDPM':
        model = Model(config)
        states = torch.load(egsde.args.ckpt)
        model = model.to(egsde.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states, strict=True)
        model.eval()

    # load domain-specific feature extractor
    dse = create_dse(image_size=config.data.image_size,
                     num_class=config.dse.num_class,
                     classifier_use_fp16=config.dse.classifier_use_fp16,
                     classifier_width=config.dse.classifier_width,
                     classifier_depth=config.dse.classifier_depth,
                     classifier_attention_resolutions=config.dse.classifier_attention_resolutions,
                     classifier_use_scale_shift_norm=config.dse.classifier_use_scale_shift_norm,
                     classifier_resblock_updown=config.dse.classifier_resblock_updown,
                     classifier_pool=config.dse.classifier_pool,
                     phase=args.phase)
    states = torch.load(args.dsepath)
    dse.load_state_dict(states)
    dse.to(egsde.device)
    dse = torch.nn.DataParallel(dse)
    dse.eval()

    # load domain-independent feature extractor
    shape = (args.batch_size, 3, config.data.image_size, config.data.image_size)
    shape_d = (
        args.batch_size, 3, int(config.data.image_size / args.down_N), int(config.data.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(egsde.device)
    up = Resizer(shape_d, args.down_N).to(egsde.device)
    die = (down, up)

    # create dataset
    dataset = get_dataset(phase=args.phase, image_size=config.data.image_size, data_path=args.testdata_path)
    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    #



    for i, (y, name) in enumerate(data_loader):
        logging.info(f'batch:{i}/{len(dataset) / args.batch_size}')
        n = y.size(0)
        y0 = rescale(y).to(egsde.device)
        #let x0 be source image
        x0 = y0
        # original.append(x0)
        #args.sample_step: the times for repeating EGSDE(usually set 1) (see Appendix A.2)
        for it in range(args.sample_step):
            e = torch.randn_like(y0)
            total_noise_levels = args.t
            a = (1 - egsde.betas).cumprod(dim=0)
            # the start point M: y ∼ qM|0(y|x0)
            y = y0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            for i in tqdm(reversed(range(total_noise_levels))):
                t = (torch.ones(n) * i).to(egsde.device)
                #sample perturbed source image from the perturbation kernel: x ∼ qs|0(x|x0)
                xt = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                # egsde update (see VP-EGSDE in Appendix A.3)
                y_ = egsde_sample(y=y, dse=dse,ls=args.ls,die=die,li=args.li,t=t,model=model,
                                    logvar=egsde.logvar,betas=egsde.betas,xt=xt,s1=args.s1,s2=args.s2, model_name = args.diffusionmodel)
                y = y_
            y0 = y  #20x3x256x256
            y = inverse_rescale(y)
            # edited.append(y)

            # save image
            for b in range(n):
                os.makedirs(sample_dir, exist_ok=True)  #instead of egsde.args.samplepath
                path_name= os.path.join(sample_dir, name[b])
                tvu.save_image(
                    y[b], path_name
                )





    logging.info('Finshed sampling.')



    # export PYTHONPATH="${PYTHONPATH}:/local/scratch/a/rahman89/PycharmProjects/EGSDE"
