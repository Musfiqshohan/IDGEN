from cfg.unet import Unet
from ncm_mnist.ModularUtils.ControllerConstants import map_dictfill_to_discrete, map_fill_to_discrete, get_multiple_labels_fill, init_weights
from ncm_mnist.ModularUtils.Discriminators import DigitImageDiscriminator
from ncm_mnist.ModularUtils.Generators import DigitImageGenerator, \
    gumbel_softmax, ConditionalClassifier, UnetGenerator
import torch
from pathlib import Path
from numpy import uint8

from torch import optim as optim
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_mnist.ModularUtils.FunctionsDistribution import get_joint_distributions_from_samples
from ncm_mnist.mnist_unet import build_unet


# mish activation function
class mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        mish()
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        mish(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        mish()
    )


class conditionalMobileNet(nn.Module):
    def __init__(self, **kwargs):
        super(conditionalMobileNet, self).__init__()

        input_dim= 3+ kwargs['noise_dim']
        output_dim = kwargs['output_dim']

        # num_labels = 3
        self.features = nn.Sequential(
            conv_bn(input_dim, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )

        self.fc = nn.Linear(1024, output_dim)



    def forward(self, Exp, noises, x, **kwargs):
        noises = torch.cat(noises, 1)
        x = torch.cat(x, 1)
        x = torch.cat([noises,x], 1)

        x= torch.cat(x, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        output_feature = gumbel_softmax(Exp, x, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)
        return output_feature

# from ModularUtils.imageVae import DeepAutoencoder


def get_generators(Exp, load_which_models):
    label_generators = {}
    optimizersMech = {}

    for label in Exp.Observed_DAG:

        noise_dims = Exp.NOISE_DIM + Exp.CONF_NOISE_DIM * len(
            Exp.latent_conf[label])

        parent_dims = 0
        for par in Exp.Observed_DAG[label]:
            parent_dims += Exp.label_dim[par]



        # Generator for W1
        if label =='W1':
            input_nc = Exp.IMAGE_NOISE_DIM + Exp.CONF_NOISE_DIM * len(Exp.latent_conf[label]) + parent_dims
            label_generators[label] =build_unet(input_nc).to(Exp.DEVICE)
            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas, weight_decay=Exp.generator_decay)

        if label =='W2a':  #Need noise for classifier.
            label_generators[label] = ConditionalClassifier(noise_dim=3, parent_dims=3, output_dim=Exp.label_dim[label]).to(Exp.DEVICE) # image parent.
            momentum = 0.5
            optimizersMech[label] = optim.SGD(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                  momentum=momentum)

        if label =='W2b':  #Need noise for classifier.
            label_generators[label] = ConditionalClassifier(noise_dim=3, parent_dims=3, output_dim=Exp.label_dim[label]).to(Exp.DEVICE) # image parent.
            momentum = 0.5
            optimizersMech[label] = optim.SGD(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                  momentum=momentum)

        if label == 'X':
            input_nc = Exp.IMAGE_NOISE_DIM + Exp.CONF_NOISE_DIM * len(Exp.latent_conf[label])+parent_dims
            label_generators[label] =build_unet(input_nc).to(Exp.DEVICE)
            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas, weight_decay=Exp.generator_decay)

        if label=='Y':
            input_nc = Exp.IMAGE_NOISE_DIM + Exp.CONF_NOISE_DIM * len(Exp.latent_conf[label]) + parent_dims
            label_generators[label] =build_unet(input_nc).to(Exp.DEVICE)

            # generator = UnetGenerator(input_nc, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False).cuda().float()
            # # init_weights(generator, 'normal', scaling=0.02)
            # generator = torch.nn.DataParallel(generator)  # multi-GPUs
            # label_generators[label] = generator

            # label_generators[label] = DigitImageGenerator(noise_dim=Exp.IMAGE_NOISE_DIM+Exp.CONF_NOISE_DIM * len(Exp.latent_conf[label]),
            #                                               parent_dims=3,
            #                                               num_filters=Exp.IMAGE_FILTERS,
            #                                               output_dim=3).to(Exp.DEVICE)  # mnistImage

            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas, weight_decay=Exp.generator_decay)





        label_generators[label].apply(init_weights)


    return label_generators, optimizersMech


def get_discriminators(Exp):

    critic={}
    optimizer={}

    critic['ncm_joint'] = DigitImageDiscriminator(
        image_dim=3+10+6+3+3,  #W1=3, W2a=10, W2b=6, X=3, Y=3
        num_filters=Exp.IMAGE_FILTERS[::-1],
        output_dim=1
    ).to(Exp.DEVICE)

    for key in critic:
        optimizer[key]= torch.optim.Adam(critic[key].parameters(), lr=Exp.learning_rate, betas=Exp.betas,
                         weight_decay=Exp.discriminator_decay)

    return critic, optimizer


def get_generated_labels(Exp, label_generators,  intervened, chosen_labels, mini_batch, **kwargs):
    label_noises={}
    for name in Exp.label_names:
        if name not in Exp.image_labels:
            label_noises[Exp.exogenous[name]] = torch.randn(mini_batch, Exp.NOISE_DIM).to(
                Exp.DEVICE)  # white noise. no bias

    conf_noises={}
    for label in Exp.label_names:
        confounders = Exp.latent_conf[label]
        for conf in confounders:  # no confounder name, only their sequence matters here.
            conf_noises[conf] = torch.randn(mini_batch, Exp.CONF_NOISE_DIM).to(Exp.DEVICE)  # white noise. no bias

    max_in_top_order = max([Exp.label_names.index(lb) for lb in chosen_labels])
    # print("max_in_top_order", max_in_top_order)
    gen_labels = {}
    for lbid, label in enumerate(Exp.Observed_DAG):
        if lbid > max_in_top_order:  # we dont need to produce the rest of the variables.
            break

        # print(lbid, label)
        Noises = []
        if label not in Exp.image_labels:
            Noises.append(label_noises[Exp.exogenous[label]])  # error here

        for conf in Exp.latent_conf[label]:
            Noises.append(conf_noises[conf])


        # getting observed parent values
        parent_gen_labels = []
        for parent in Exp.Observed_DAG[label]:
            parent_gen_labels.append(gen_labels[parent])

        if label in intervened.keys():
            if torch.is_tensor(intervened[label]):
                gen_labels[label] = intervened[label]
            else:
                gen_labels[label] = torch.ones(mini_batch, Exp.label_dim[label]).to(Exp.DEVICE) * 0.00001
                gen_labels[label][:, intervened[label]] = 0.99999

        elif label in Exp.image_labels:

            Noises = []
            image_noise = torch.randn(mini_batch, Exp.IMAGE_NOISE_DIM).view(-1, Exp.IMAGE_NOISE_DIM, 1, 1).to(
                Exp.DEVICE)
            Noises.append(image_noise)
            for conf in Exp.latent_conf[label]:
                Noises.append(conf_noises[conf].view(-1, Exp.CONF_NOISE_DIM, 1, 1).to(Exp.DEVICE))


            parent_gen_labels = torch.cat(parent_gen_labels, 1)
            dims_list = [Exp.label_dim[lb] for lb in Exp.Observed_DAG[label]]
            parent_gen_labels = map_fill_to_discrete(Exp, parent_gen_labels, dims_list)
            parent_gen_labels = [get_multiple_labels_fill(Exp, parent_gen_labels, dims_list, isImage_labels=True, more_dimsize=1)]

            gen_labels[label] = label_generators[label](Noises, parent_gen_labels) #sending lists


        elif set(Exp.Observed_DAG[label]) & set(Exp.image_labels) != set():
            for idx, par_label in enumerate(parent_gen_labels):
                if len(par_label.shape)<4:
                    parent_gen_labels[idx]= par_label.unsqueeze(2).unsqueeze(3).repeat(1, 1, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE)
            gen_labels[label] = label_generators[label](Exp,Noises, parent_gen_labels, gumbel_noise=None, hard=False)


    return_labels = {}
    for label in chosen_labels:
        return_labels[label] = gen_labels[label]

    return return_labels




def get_fake_distribution(Exp, label_generators, intv_key, compare_Var ):
    generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, dict(intv_key), compare_Var,
                                                 Exp.Synthetic_Sample_Size)
    generated_labels_full = map_dictfill_to_discrete(Exp, generated_labels_dict, compare_Var)
    fake_dist_dict = get_joint_distributions_from_samples(Exp, compare_Var, generated_labels_full, "feature")

    return fake_dist_dict