import os
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



def get_prediction(classifier, trainer, images):


    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]


    transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )

    data_list = []
    for img in images:
        lbl = torch.zeros(40, 1)
        data_list.append([transform(img), lbl])

    predict_loader = DataLoader(dataset=data_list, batch_size=1, shuffle=False)
    prediction = trainer.predict(classifier, predict_loader)  # without fine-tuning
    all=[]
    for idx, data_input in enumerate(prediction):
        pred= data_input[2][0]
        all.append(pred)
    return all



def plot_image_ara(img_ara, folder=None, title=None):
    rows=img_ara.shape[0]
    cols=img_ara.shape[1]

    print(rows,cols)

    f, axarr = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for c in range(cols):

        for r in range(rows):
            axarr[r, c].get_xaxis().set_ticks([])
            axarr[r, c].get_yaxis().set_ticks([])

            img= img_ara[r][c].cpu().detach().numpy()
            img= np.transpose(img, (1,2,0))
            axarr[r, c].imshow(img)


        f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if folder==None:
        plt.show()
    else:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{title}.png', bbox_inches='tight')

    plt.close()