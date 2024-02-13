from PIL import Image
import glob
import torchvision
from matplotlib import pyplot as plt
import os

from Classifier.pre_trained import get_classifier
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


from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from constantFunctions import get_prediction, plot_image_ara

import argparse
argsall = argparse.Namespace(
testdata_path='/local/scratch/a/rahman89/PycharmProjects/EGSDE/data/celeba_hq/val/male',
ckpt = '/local/scratch/a/rahman89/PycharmProjects/EGSDE/pretrained_model/celebahq_female_ddpm.pth',
dsepath = '/local/scratch/a/rahman89/PycharmProjects/EGSDE/pretrained_model/male2female_dse.pt',
config_path = '/local/scratch/a/rahman89/PycharmProjects/EGSDE/profiles/male2female/male2female.yml',
t = 500,
ls =  500.0,
li = 2.0,
s1 = 'cosine',
s2 = 'neg_l2',
phase = 'test',
root = 'runs/',
sample_step= 1,
batch_size = 20,
diffusionmodel = 'DDPM',
down_N = 32,
seed=1234)




task = 'male2female'
# from profiles.male2female.args import argsall

# args
args = argsall
set_seed(args.seed)
args.samplepath = os.path.join('runs', task)
os.makedirs(args.samplepath, exist_ok=True)
set_logger(args.samplepath, 'sample.txt')

#config
with open(args.config_path, "r") as f:
    config_ = yaml.safe_load(f)
config = dict2namespace(config_)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
egsde = EGSDE(args, config)

args, config = egsde.args, egsde.config


transform = transforms.Compose([
                                # transforms.PILToTensor(),
                                transforms.ToTensor(),
                                transforms.Resize((512,512))])



image_list = []
for filename in glob.glob(f'{args.testdata_path}/*.png'): #assuming gif
    im=Image.open(filename)
    im= transform(im).to(egsde.device)
    image_list.append(im.unsqueeze(0))

origina_images= image_list[0:24]
origina_images = torch.cat(origina_images)

# grid_img = torchvision.utils.make_grid(origina_images, nrow=4)
# fig = plt.figure()

# plt.imshow(grid_img.cpu().permute(1, 2, 0))
# plt.savefig('/local/scratch/a/rahman89/PycharmProjects/EGSDE/IDDAG/PLOTS/original.pdf')

# plt.show()



image_list = []
for filename in glob.glob('/local/scratch/a/rahman89/PycharmProjects/EGSDE/runs/male2female/0/*.png'): #assuming gif
    im=Image.open(filename)
    im= transform(im).to(egsde.device)
    image_list.append(im.unsqueeze(0))

edited_images= image_list[0:24]
edited_images = torch.cat(edited_images)

all_img= torch.cat([origina_images[0:12].unsqueeze(0), edited_images[0:12].unsqueeze(0)])
plot_image_ara(all_img, '/local/scratch/a/rahman89/PycharmProjects/EGSDE/IDDAG/PLOTS/','translation1')

all_img= torch.cat([origina_images[12:24].unsqueeze(0), edited_images[12:24].unsqueeze(0)])
plot_image_ara(all_img, '/local/scratch/a/rahman89/PycharmProjects/EGSDE/IDDAG/PLOTS/','translation2')


label_path = "/local/scratch/a/rahman89/Datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
attributes = open(label_path).readlines()[1].split(' ')
attributes[-1] = attributes[-1].strip('\n')
classifier, trainer = get_classifier(attributes, IMAGE_SIZE=128)

####

reduced, increased = {}, {}
for lb in attributes:
    reduced[lb] = 0
    increased[lb] = 0`

pred1 = get_prediction(classifier, trainer, origina_images)
pred2 = get_prediction(classifier, trainer, edited_images)

for st, en in zip(pred1, pred2):

    diff = set(st) - set(en)
    for lb in diff:
        reduced[lb] += 1

    diff = set(en) - set(st)
    for lb in diff:
        increased[lb] += 1



print('Attribute reduced')
print(reduced)
for lb in reduced:
    reduced[lb] = reduced[lb]/ (origina_images.shape[0])*100
reduced =dict(sorted(reduced.items(), key=lambda item: item[1]))
print(reduced)

print('Attribute increased')
print(increased)
for lb in increased:
    increased[lb] = increased[lb]/ (origina_images.shape[0])*100
increased =dict(sorted(increased.items(), key=lambda item: item[1]))
print(increased)
