# CelebA image generation using Conditional DCGAN
import os
import pickle
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from ModularUtils.FunctionsConstant import load_label_dataset, asKey, initialize_results, load_image_dataset
from ModularUtils.ControllerConstants import get_multiple_labels_fill, fill2d_to_fill4d, map_fill_to_discrete, \
    map_dictfill_to_discrete
from ModularUtils.ControllerModel import get_generators, get_discriminators
from ModularUtils.Experiment_Class import Experiment
from ModularUtils.FunctionsTraining import get_training_variables, labels_image_gradient_penalty, calc_gradient_penalty, \
    save_checkpoint, image_gradient_penalty
from ncm_mnist.ModularUtils.FunctionsDistribution import get_joint_distributions_from_samples, calculate_TVD, \
    calculate_KL

from ncm_mnist.napkin_graph import set_napkin

def get_generated_labels(Exp, label_generators,  intervened, chosen_labels, mini_batch, **kwargs):

    conf_noises={}
    for label in Exp.label_names:
        confounders = Exp.latent_conf[label]
        for conf in confounders:  # no confounder name, only their sequence matters here.
            conf_noises[conf] = torch.randn(mini_batch, Exp.CONF_NOISE_DIM, Exp.IMAGE_SIZE,Exp.IMAGE_SIZE).to(Exp.DEVICE)  # white noise. no bias

    max_in_top_order = max([Exp.label_names.index(lb) for lb in chosen_labels])
    gen_labels = {}
    for lbid, label in enumerate(Exp.Observed_DAG):
        if lbid > max_in_top_order:  # we dont need to produce the rest of the variables.
            break

        # print(lbid, label)
        Noises = []
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
            image_noise = torch.randn(mini_batch, Exp.IMAGE_NOISE_DIM,Exp.IMAGE_SIZE, Exp.IMAGE_SIZE).to(Exp.DEVICE)
            # image_noise = torch.randn(mini_batch, Exp.IMAGE_NOISE_DIM,256, 256).to(Exp.DEVICE)
            Noises.append(image_noise)
            for conf in Exp.latent_conf[label]:
                Noises.append(conf_noises[conf].to(Exp.DEVICE))

            # if len(parent_gen_labels)!=0:
            #     for idx in range(len(parent_gen_labels)):
            #         if len(parent_gen_labels[idx].shape)==4:
            #             continue
            if label=='X': #X has two discrete labels that needs to be the same size as image
                parent_gen_labels = torch.cat(parent_gen_labels, 1)
                dims_list = [Exp.label_dim[lb] for lb in Exp.Observed_DAG[label]]
                parent_gen_labels = map_fill_to_discrete(Exp, parent_gen_labels, dims_list)
                parent_gen_labels = [get_multiple_labels_fill(Exp, parent_gen_labels, dims_list, isImage_labels=True, image_size=Exp.IMAGE_SIZE)]


            input= torch.cat(Noises+ parent_gen_labels, dim=1)
            # input= Noises[0]
            gen_labels[label] = label_generators[label](input) #sending lists


        elif set(Exp.Observed_DAG[label]) & set(Exp.image_labels) != set():
            # for idx, par_label in enumerate(parent_gen_labels):
            #     if len(par_label.shape)<4:
            #         parent_gen_labels[idx]= par_label.unsqueeze(2).unsqueeze(3).repeat(1, 1, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE)

            newNoise = [torch.randn(mini_batch, 3 , Exp.IMAGE_SIZE, Exp.IMAGE_SIZE).to(Exp.DEVICE)]  #noise same size  as image
            gen_labels[label] = label_generators[label](Exp,newNoise, parent_gen_labels, gumbel_noise=None, hard=False)


    return_labels = {}
    for label in chosen_labels:
        return_labels[label] = gen_labels[label]

    return return_labels

def train_batch(Exp, cur_mechs, label_generators, G_optimizers, critic, D_optimizer,
                           label_data, image_data,):

    dims_list = [10, 6] #W2a, W2b

    real_W2 = torch.cat(list(label_data.values()), dim=1)
    real_W2 = get_multiple_labels_fill(Exp, real_W2, dims_list, isImage_labels=True, image_size= Exp.IMAGE_SIZE)  # 10 ->
    mini_batch = real_W2.shape[0]
    real_images = torch.cat(list(image_data.values()), dim=1)
    real_joint= torch.cat([real_W2, real_images], dim=1)  #W2a,W2b, W1,X,Y ~ P(W2a,W2b,W1,X,Y)  = 10+6+3+3+3=25


    gen_samples = get_generated_labels(Exp, label_generators,  {}, Exp.label_names, mini_batch)
    fake_W2=  torch.cat([gen_samples['W2a'], gen_samples['W2b']], dim=1)
    fake_W2 = fill2d_to_fill4d(Exp, fake_W2, image_size=Exp.IMAGE_SIZE)
    fake_images = torch.cat([gen_samples['W1'], gen_samples['X'], gen_samples['Y']], dim=1)
    fake_joint= torch.cat([fake_W2, fake_images], dim=1)  #W2a,W2b, W1,X,Y ~ P(W2a,W2b,W1,X,Y)

    D_losses = []
    for crit_ in range(Exp.CRITIC_ITERATIONS):
        # P(W2a, W2b, W1, X, Y)
        D_real_joint = critic['ncm_joint'](real_joint).squeeze()
        D_fake_joint = critic['ncm_joint'](fake_joint).squeeze()
        gp_joint = image_gradient_penalty(critic['ncm_joint'], real_joint, fake_joint, device=Exp.DEVICE)
        D_loss_joint = (-  (torch.mean(D_real_joint) - torch.mean(D_fake_joint)) + Exp.LAMBDA_GP * gp_joint)
        critic['ncm_joint'].zero_grad()
        D_loss_joint.backward(retain_graph=True)
        D_optimizer['ncm_joint'].step()
        D_losses.append((D_loss_joint).data)  # list of critic loss

    #%%%%%%%%%%%%%%%%%%% generator  training  %%%%%%%%%%%%%%%%%%%
    for mech in cur_mechs:
        label_generators[mech].zero_grad()

    gen_samples = get_generated_labels(Exp, label_generators, {}, Exp.label_names, mini_batch)
    fake_W2 = torch.cat([gen_samples['W2a'], gen_samples['W2b']], dim=1)
    fake_W2 = fill2d_to_fill4d(Exp, fake_W2, image_size=Exp.IMAGE_SIZE)
    fake_images = torch.cat([gen_samples['W1'], gen_samples['X'], gen_samples['Y']], dim=1)
    fake_joint = torch.cat([fake_W2, fake_images], dim=1)  # W2a,W2b, W1,X,Y ~ P(W2a,W2b,W1,X,Y)

    D_fake_joint = critic['ncm_joint'](fake_joint).squeeze()
    G_loss = -torch.mean(D_fake_joint)

    # Back propagation
    G_loss.backward()
    for mech in cur_mechs:
        G_optimizers[mech].step()

    D_loss = torch.mean(torch.FloatTensor(D_losses))  # just mean of losses
    return G_loss.data, D_loss.data, gen_samples



def labelMain(Exp, cur_hnodes, label_generators, G_optimizers, discriminators, D_optimizers, label_data, image_data ,TVD, KL):

    W2a_loader = torch.utils.data.DataLoader(dataset=label_data['W2a'], batch_size=Exp.batch_size, shuffle=False)
    W2b_loader = torch.utils.data.DataLoader(dataset=label_data['W2b'], batch_size=Exp.batch_size, shuffle=False)

    label_batch = []
    for W2a_batch, W2b_batch in zip(W2a_loader, W2b_loader):
        label_batch.append({'W2a': W2a_batch.view(-1,1), 'W2b':W2b_batch.view(-1,1) })

    W1_loader = torch.utils.data.DataLoader(dataset=image_data["W1"],batch_size=Exp.batch_size, shuffle=False)
    X_loader = torch.utils.data.DataLoader(dataset=image_data["X"],batch_size=Exp.batch_size,shuffle=False)
    Y_loader = torch.utils.data.DataLoader(dataset=image_data["Y"],batch_size=Exp.batch_size,shuffle=False)

    image_batches = []
    for W1_batch,X_batch,Y_batch in zip(W1_loader, X_loader, Y_loader):
        # data_input = torch.squeeze(data_input)
        image_batches.append({'W1':W1_batch, 'X':X_batch, 'Y':Y_batch})


    iteration = 0
    num_batches = len(label_batch)

    tvd =[]
    kl =[]
    for batchno in range(num_batches):
        label_data = label_batch[batchno]
        g_loss, d_loss, gen_samples = train_batch(Exp, cur_hnodes['H1'], label_generators, G_optimizers, discriminators, D_optimizers, label_data , image_batches[batchno])


        compare_Var = ['W2a', 'W2b']
        generated_labels_full = map_dictfill_to_discrete(Exp, gen_samples, compare_Var)
        fake_dist_dict = get_joint_distributions_from_samples(Exp, compare_Var, generated_labels_full)
        real_data= torch.cat(list(label_data.values()), dim=1).detach().cpu().numpy().astype(int)
        dataset_dist_dict = get_joint_distributions_from_samples(Exp, compare_Var, real_data)
        obs_tvd = calculate_TVD(fake_dist_dict, dataset_dist_dict, doPrint=False)
        obs_kl = calculate_KL(fake_dist_dict, dataset_dist_dict, doPrint=False)
        tvd.append(obs_tvd)
        kl.append(obs_kl)

        print('Epoch [%d/%d], Step [%d/%d],' % (Exp.curr_epoochs + 1, Exp.num_epochs, iteration + 1, num_batches),
          'mechanism: ',cur_hnodes['H1'],  ' D_loss: %.4f, G_loss: %.4f' % (d_loss.data, g_loss.data), 'obs_tvd=%.4f,  obs_kl;%.4f'%(obs_tvd, obs_kl))
        print(f'W2a:{generated_labels_full[0:20,0]} W2b: {generated_labels_full[0:20,1]}')


        # Annealing
        tot_iter = Exp.curr_epoochs * num_batches + iteration
        if tot_iter % 100 == 0:
            Exp.anneal_temperature(tot_iter)


        iteration += 1



    #

#
    if (Exp.curr_epoochs) % 5 == 0:
        TVD.append(np.mean(tvd))
        KL.append(np.mean(kl))

        samples = {}
        randices = random.sample(range(1, Exp.batch_size), 20)
        for lb in gen_samples:
            samples[lb] = gen_samples[lb][randices]

        os.makedirs(f'./{Exp.SAVED_PATH}/samples', exist_ok=True)
        with open(f'./{Exp.SAVED_PATH}/samples/Jointepoch{Exp.curr_epoochs}.pkl', 'wb') as pkl_file:
            pickle.dump(samples, pkl_file)

        with open(f'./{Exp.SAVED_PATH}/samples/TVDepoch{Exp.curr_epoochs}.pkl', 'wb') as pkl_file:
            pickle.dump(TVD, pkl_file)

        with open(f'./{Exp.SAVED_PATH}/samples/KLepoch{Exp.curr_epoochs}.pkl', 'wb') as pkl_file:
            pickle.dump(KL, pkl_file)

        save_checkpoint(Exp, Exp.SAVED_PATH, label_generators, G_optimizers, discriminators, D_optimizers)
        print(Exp.curr_epoochs,":model saved at ", Exp.SAVED_PATH)

    return




if __name__ == "__main__":

    args = sys.argv

    if len(args) == 1:
        exp_name = 'ncmMNIST'
    else:
        exp_name = args[1]

    Exp = Experiment(set_napkin,
                     exp_name=exp_name,
                     Temperature=1,
                     temp_min=0.1,
                     learning_rate=5 * 1e-4,
                     batch_size=200,
                     IMAGE_NOISE_DIM=3,
                     CONF_NOISE_DIM=3,
                     ENCODED_DIM=10,
                     Data_intervs=[{}],
                     num_epochs=301,
                     new_experiment=True
                     )


    print('Temperature:',Exp.Temperature, 'temp_min',Exp.temp_min)

    os.makedirs(Exp.SAVED_PATH, exist_ok=True)

    dag_name = Exp.Complete_DAG_desc + ".txt"

    # Exp.LOAD_MODEL_PATH = "/path_to_project/SAVED_EXPERIMENTS/imageMediator/Exp1/Mar_17_2023-14_40"
    Exp.load_which_models = ['rW1', 'rX', 'rY']

    cur_hnodes = {"H1":["W2a", "W2b", "W1","X", "Y"]}



    Exp.LAMBDA_GP=10
    label_generators, optimizersMech = get_generators(Exp, Exp.load_which_models)
    discriminatorsMech, doptimizersMech = get_discriminators(Exp)  #


    file = '../napkin_mnist/base_data/napkin_mnist_train.pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)


    image_data =  {}
    for lb in Exp.image_labels:
        image_data[lb] = torch.tensor(data[lb]).to(Exp.DEVICE)

    label_data= {'W2a':torch.tensor(data['W2a']).to(Exp.DEVICE), 'W2b':torch.tensor(data['W2b']).to(Exp.DEVICE)}


    mech_tvd = 0
    print("Starting training new mechanism")

    TVD=[]
    KL =[]

    for epoch in tqdm(range(Exp.num_epochs)):
        Exp.curr_epoochs = epoch
        labelMain(Exp, cur_hnodes, label_generators, optimizersMech, discriminatorsMech, doptimizersMech, label_data, image_data ,TVD, KL)



