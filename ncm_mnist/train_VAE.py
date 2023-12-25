import os
import pickle

import torch
import random
from tqdm import tqdm


import torch.nn as nn
from torch import optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F

from ModularUtils.FunctionsConstant import asKey
from ModularUtils.ControllerConstants import init_weights
from ModularUtils.DigitImageGeneration.mnist_image_generation import plot_trained_digits
from ncm_mnist.ModularUtils.Experiment_Class import Experiment
from ncm_mnist.napkin_graph import set_napkin


# Creating a DeepAutoencoder class
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
class DeepAutoencoder(torch.nn.Module):
    def __init__(self, Exp, par_dim, latent_dim):
        super().__init__()

        label_dim = par_dim
        num_input_channels=3
        base_channel_size=Exp.IMAGE_SIZE
        c_hid = base_channel_size
        latent_dim=latent_dim
        act_fn: object = nn.GELU

        # for 32
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels+label_dim, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

        self.linear = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 2 * 16 * c_hid),   #latent dim -> 32*32= 1024
            act_fn()
        )


    def forward(self, x, isLatent=True):
            x = torch.cat([x], 1)
            z = self.encoder(x)

            if isLatent:
                return z

            z = torch.cat([z], 1)
            z = self.linear(z)
            z = z.reshape(z.shape[0], -1, 4, 4)
            x_hat = self.decoder(z)
            return x_hat





def train_encoders(Exp, rep_mech , label_generators, optimizers, image_data):
    criterion = torch.nn.MSELoss()
    num_epochs = 500

    image_data_loader = torch.utils.data.DataLoader(dataset=image_data,
                                                    batch_size=Exp.batch_size,
                                                    shuffle=False)



    train_loss = []
    outputs = {}

    batch_size = len(image_data_loader)

    for epoch in range(num_epochs):
        running_loss = 0



        # Iterating over the training dataset
        for bno, batch in enumerate (tqdm(image_data_loader)):
            if type(batch) is list:
                img = batch[0].to(Exp.DEVICE)
            else:
                img= batch.to(Exp.DEVICE)

            out = label_generators[rep_mech](img, isLatent=False)

            loss = criterion(out, img)

            optimizers[rep_mech].zero_grad()
            loss.backward()
            optimizers[rep_mech].step()

            running_loss += loss.item()


        # Averaging out loss over entire batch
        running_loss /= batch_size
        train_loss.append(round(running_loss,4))

        ll = -min(10, len(train_loss))
        print("epoch:", epoch, train_loss[ll:])

        outputs[epoch + 1] = {'img': img, 'out': out}


        if epoch%25==0:
            rind= random.randint(0, img.shape[0])
            img = img[rind].permute(1, 2, 0).detach().cpu().numpy()
            plot_trained_digits(1, 1, [img], f'Epoch-{epoch}-Real', Exp.SAVED_PATH+f"/VAE_plots/{rep_mech}")

            out = out[rind].permute(1, 2, 0).detach().cpu().numpy()
            plot_trained_digits(1, 1, [out], f'Epoch-{epoch}-fake',Exp.SAVED_PATH+f"/VAE_plots/{rep_mech}")

            # saving models
            print(Exp.curr_epoochs, ":Encoder model saved at ", Exp.SAVED_PATH)
            print("=> Saving checkpoint")
            gen_checkpoint = {"epoch": Exp.curr_epoochs,
            "state_dict" : label_generators[rep_mech].state_dict(),
            "optimizer":  optimizers[rep_mech].state_dict()
            }

            os.makedirs(Exp.SAVED_PATH + f"/checkpoints/encoder{rep_mech}", exist_ok=True)
            gfile = Exp.SAVED_PATH + f"/checkpoints/encoder{rep_mech}/epoch{Exp.curr_epoochs:03}.pth"
            last_gfile = Exp.SAVED_PATH + f"/checkpoints/encoder{rep_mech}/epochLast.pth"
            torch.save(gen_checkpoint, gfile)
            torch.save(gen_checkpoint, last_gfile)





    # Plotting the training loss
    plt.plot(range(0, epoch + 1), train_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Loss")
    plt.savefig(f'{Exp.SAVED_PATH}/VAE_plots/{rep_mech}/loss.png', bbox_inches='tight')
    print('done')




if __name__ == '__main__':
    Exp = Experiment(set_napkin,
                     learning_rate=5 * 1e-4,
                     Synthetic_Sample_Size=60000,
                     batch_size=200,
                     ENCODED_DIM=10,
                     Data_intervs=[{}],
                     num_epochs=300,
                     new_experiment=True
                     )

    os.makedirs(Exp.SAVED_PATH, exist_ok=True)

    dag_name = Exp.Complete_DAG_desc + ".txt"

    # Exp.load_which_models = {"W1": False, "X": False, "Y": False}
    Exp.load_which_models = []

    # %%%%%%%
    label_generators = {}
    optimizersMech = {}
    for label in Exp.rep_labels:

        if label in Exp.load_which_models:
            continue

        noise_dims = Exp.NOISE_DIM + Exp.CONF_NOISE_DIM * len(
            Exp.latent_conf[label])

        parent_dims = 0

        label_generators[label] = DeepAutoencoder(Exp, parent_dims, latent_dim=Exp.ENCODED_DIM).to(Exp.DEVICE)

        optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                 betas=Exp.betas, weight_decay=Exp.generator_decay)

        label_generators[label].apply(init_weights)



    #
    file = '../napkin_mnist/base_data/napkin_mnist_train.pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)



    # for img_lbl, rep_lbl in zip(Exp.image_labels,Exp.rep_labels):

    idx= 2
    img_lbl, rep_lbl= Exp.image_labels[idx] ,Exp.rep_labels[idx]
    image_data =  data[img_lbl]
    print('Training:',rep_lbl)
    train_encoders(Exp, rep_lbl, label_generators, optimizersMech, image_data)
