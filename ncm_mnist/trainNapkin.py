# CelebA image generation using Conditional DCGAN
import os
import pickle

import torch
from tqdm import tqdm

from ModularUtils.FunctionsConstant import load_label_dataset, asKey, initialize_results, load_image_dataset
from ModularUtils.ControllerConstants import get_multiple_labels_fill, fill2d_to_fill4d
from ModularUtils.ControllerModel import get_generators, get_discriminators, get_generated_labels
from ModularUtils.Experiment_Class import Experiment
from ModularUtils.FunctionsTraining import get_training_variables, labels_image_gradient_penalty, calc_gradient_penalty, \
    save_checkpoint

from ncm_mnist.napkin_graph import set_napkin


def train_ImgMech(Exp, cur_mechs, label_generators, G_optimizers, critic, D_optimizer,
                           label_data, image_data,):

    dims_list = [10, 6] #W2a, W2b
    real_W2 = torch.cat(label_data.keys(), 1)
    real_W2 = get_multiple_labels_fill(Exp, real_W2, dims_list, isImage_labels=False)
    mini_batch = real_W2.shape[0]


    real_labels_forImg = get_multiple_labels_fill(Exp, current_real_label, dims_list, isImage_labels=True, more_dimsize=Exp.IMAGE_SIZE)
    obs_images = image_data

    intv_tensor_dict = {}
    generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, intv_tensor_dict,
                                                 Exp.label_names, mini_batch)



    generated_image=None
    if set(cur_mechs) & set(Exp.image_labels) != set():
        generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, intv_tensor_dict,
                                                     intervened_Var + all_compare_Var, mini_batch, hard=True)
        generated_image = generated_labels_dict[Exp.image_labels[0]]
        del generated_labels_dict[Exp.image_labels[0]]
        y_dims = sum([Exp.label_dim[lb] for lb in real_labels_vars])
        ret = list(generated_labels_dict.values())
        ret2d = torch.cat(ret, 1).view(-1, y_dims) #for critic
        generated_labels_fill= fill2d_to_fill4d(Exp, ret2d, more_dimsize=Exp.IMAGE_SIZE)

    elif set(all_compare_Var) & set(Exp.rep_labels) != set():
        generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, intv_tensor_dict,real_labels_vars+Exp.rep_labels, mini_batch)
        y_dims = sum([Exp.label_dim[lb] for lb in real_labels_vars+Exp.rep_labels])
        ret = list(generated_labels_dict.values())
        generated_labels_fill = torch.cat(ret, 1).view(-1, y_dims)
    else:
        generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, intv_tensor_dict, real_labels_vars, mini_batch)
        y_dims = sum([Exp.label_dim[lb] for lb in real_labels_vars])
        ret = list(generated_labels_dict.values())
        generated_labels_fill = torch.cat(ret, 1).view(-1, y_dims)



    #%%%%%%%%%%%%%%%%%%% critic training  %%%%%%%%%%%%%%%%%%%
    # for crit_ in range(Exp.CRITIC_ITERATIONS):
    # P(W1,W2,X,Y)

    rW1= label_generators['rW1'](image_data['W1'])
    rX= label_generators['rW1'](image_data['W1'])
    rY= label_generators['rW1'](image_data['W1'])
    real_labels_fill= torch.cat([rW1, real_W2, rX, rY], 1)


    generated_labels_fill= torch.concat([generated_labels_dict['rW1'], generated_labels_dict['W2a'], generated_labels_dict['W2b'], generated_labels_dict['rX'],
                            generated_labels_dict['rY']],1)

    D_real_joint = critic['joint'](real_labels_fill).squeeze()
    D_fake_joint = critic(generated_labels_fill).squeeze()
    gp_joint = calc_gradient_penalty(critic, real_labels_fill, generated_labels_fill, device=Exp.DEVICE)
    D_loss_joint = (-  (torch.mean(D_real_joint) - torch.mean(D_fake_joint)) + Exp.LAMBDA_GP * gp_joint)
    critic['joint'].zero_grad()
    D_loss_joint.backward(retain_graph=True)
    D_optimizer['joint'].step()


    # P(W1)
    obs_images= image_data['W1']
    generated_image= generated_labels_dict['W1']
    D_real_W1 = critic['W1'](obs_images).squeeze()
    D_fake_W1 = critic['W1'](generated_image).squeeze()
    gp_W1 = labels_image_gradient_penalty(critic, obs_images, generated_image, device=Exp.DEVICE)
    L_W1 = (-  (torch.mean(D_real_W1) - torch.mean(D_fake_W1)) + Exp.LAMBDA_GP * gp_W1)
    critic['W1'].zero_grad()
    loss= L_W1+ D_loss_joint
    loss.backward(retain_graph=True)
    D_optimizer['W1'].step()

    # P(X,W2)
    obs_images= image_data['X']
    ret2d= torch.cat( [generated_labels_dict['W2a'], generated_labels_dict['W2b']] ,1)
    generated_labels_fill = fill2d_to_fill4d(Exp, ret2d, more_dimsize=Exp.IMAGE_SIZE)
    real_labels_fill= real_W2
    D_real_W2 = critic['X_W2aW2b'](obs_images, real_labels_fill).squeeze()
    D_fake_W2 = critic['X_W2aW2b'](generated_image, generated_labels_fill).squeeze()
    gp_W2 = labels_image_gradient_penalty(critic['X_W2aW2b'], obs_images, real_labels_fill, generated_image,generated_labels_fill, device=Exp.DEVICE)
    L_X = (-  (torch.mean(D_real_W2) - torch.mean(D_fake_W2)) + Exp.LAMBDA_GP * gp_W2)
    critic['X_W2aW2b'].zero_grad()
    loss = L_X + D_loss_joint
    loss.backward(retain_graph=True)
    D_optimizer['X'].step()


    # P(Y,X)['W1']
    obs_images= torch.cat([image_data['X'], image_data['Y']], 3) #along the channel dimension
    generated_image= torch.cat([generated_labels_dict['X'], generated_labels_dict['Y']], 3)

    D_real_Y = critic['Y_X'](obs_images).squeeze()
    D_fake_Y = critic['Y_X'](generated_image).squeeze()
    gp_Y = labels_image_gradient_penalty(critic['Y_X'], obs_images, generated_image, device=Exp.DEVICE)
    L_Y = (-  (torch.mean(D_real_Y) - torch.mean(D_fake_Y)) + Exp.LAMBDA_GP * gp_Y)
    critic['Y_X'].zero_grad()
    loss = L_Y + D_loss_joint
    loss.backward(retain_graph=True)
    D_optimizer['Y'].step()


    #%%%%%%%%%%%%%%%%%%% generator  training  %%%%%%%%%%%%%%%%%%%
    # Back propagation
    for mech in cur_mechs:
        label_generators[mech].zero_grad()

    G_loss = D_loss_joint + L_W1 + L_X + L_Y
    G_loss.backward()

    for mech in cur_mechs:
        G_optimizers[mech].step()

    # D_loss = torch.mean(torch.FloatTensor(D_losses))  # just mean of losses

    return G_loss.data, D_loss.data



def train_W2Mech(Exp, label_generators, G_optimizers, label_discriminator, D_optimizer, label_data, image_data):

    label_data =torch.cat(label_data.keys(), 1)
    mini_batch = label_data.size()[0]
    dims_list = [10,6]
    real_labels_fill = get_multiple_labels_fill(Exp, label_data, dims_list, isImage_labels=False)

    obs_images = image_data
    intv_parent_fill = obs_images
    intv_lb='W1'
    intv_tensor_dict = {}
    intv_tensor_dict[intv_lb] = intv_parent_fill


    generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, intv_tensor_dict, ['W2a', 'W2b'],
                                                 mini_batch)
    ret = list(generated_labels_dict.values())
    generated_labels_fill = torch.cat(ret, 1)

    D_losses = []
    for crit_ in range(Exp.CRITIC_ITERATIONS):
        D_real_decision_obs = label_discriminator(real_labels_fill).squeeze()
        D_fake_decision_obs = label_discriminator(generated_labels_fill).squeeze()
        gp_obs = calc_gradient_penalty(label_discriminator, real_labels_fill, generated_labels_fill, device=Exp.DEVICE)
        D_loss_obs = (-  (torch.mean(D_real_decision_obs) - torch.mean(D_fake_decision_obs)) + Exp.LAMBDA_GP * gp_obs)
        D_losses.append((D_loss_obs).data)  # just a loss list
        label_discriminator.zero_grad()
        D_loss_obs.backward(retain_graph=True)
        D_optimizer.step()

    D_fake_decision_obs = label_discriminator(generated_labels_fill).squeeze()
    G_loss = -torch.mean(D_fake_decision_obs)

    # Back propagation
    for mech in ['W2a', 'W2b']:
        label_generators[mech].zero_grad()

    G_loss.backward()

    for mech in ['W2a', 'W2b']:
        G_optimizers[mech].step()

    D_loss = torch.mean(torch.FloatTensor(D_losses))  # just mean of losses

    return G_loss.data, D_loss.data


def labelMain(Exp, cur_hnodes, label_generators, G_optimizers, discriminators, D_optimizers, label_data, image_data,
              tvd_diff, kl_diff):

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
    for batchno in range(num_batches):

        cur_mechs= cur_hnodes['H1']
        g_loss, d_loss   = train_W2Mech(Exp, label_generators, G_optimizers, discriminators, D_optimizers,  label_batch[batchno], image_batches[batchno])

        cur_mechs = cur_hnodes['H2']
        g_loss, d_loss= train_ImgMech(Exp, cur_mechs, label_generators, G_optimizers, discriminators,
                                                D_optimizers, label_batch[batchno], image_batches[batchno])

        print('Epoch [%d/%d], Step [%d/%d],' % (
            Exp.curr_epoochs + 1, Exp.num_epochs, iteration + 1, num_batches),
          'mechanism: ',cur_mechs,  ' D_loss: %.4f, G_loss: %.4f' % (d_loss.data, g_loss.data))



        # Annealing
        tot_iter = Exp.curr_epoochs * num_batches + iteration
        if tot_iter % 100 == 0:
            Exp.anneal_temperature(tot_iter)



        Exp.D_avg_losses.append(torch.mean(d_loss))
        Exp.G_avg_losses.append(torch.mean(g_loss))
        iteration += 1

        # break
    #
    if (Exp.curr_epoochs + 1) % 1 == 0:
        print("Turn on caffeinate or these results are gone!")
        tvd_diff, kl_diff = imageMediatorEvaluation(Exp, cur_hnodes, label_generators, dataset_dict, tvd_diff, kl_diff)
#
    # if (Exp.curr_epoochs <= 50 and (Exp.curr_epoochs + 1) % 5 == 0) or (Exp.curr_epoochs > 50 and (Exp.curr_epoochs + 1) % 15 == 0):
    if (Exp.curr_epoochs + 1) % 5 == 0:
        var_list= "".join(x for x in cur_mechs)
        save_checkpoint(Exp, Exp.SAVED_PATH, cur_mechs, label_generators, G_optimizers, {var_list:discriminators}, {var_list: D_optimizers})
        print(Exp.curr_epoochs,":model saved at ", Exp.SAVED_PATH)

    return




if __name__ == "__main__":

    Exp = Experiment(set_napkin,
                     learning_rate=5 * 1e-4,
                     Synthetic_Sample_Size=20000,
                     batch_size=200,
                     ENCODED_DIM=10,
                     Data_intervs=[{}],
                     num_epochs=300,
                     new_experiment=True
                     )



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
        image_data[lb] = data[lb]

    label_data= {'W2a':data['W2a'], 'W2b':data['W2b']}


    mech_tvd = 0
    print("Starting training new mechanism")

    for epoch in tqdm(range(Exp.num_epochs)):
        Exp.curr_epoochs = epoch
        labelMain(Exp, cur_hnodes, label_generators, optimizersMech, discriminatorsMech, doptimizersMech, label_data, image_data)


