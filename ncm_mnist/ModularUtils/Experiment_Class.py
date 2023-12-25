import itertools
import json
import os

import numpy as np
import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
# import collections
# old constants
from datetime import datetime



class Experiment:

    def __init__(self, set_truedag, **kwargs):

        self.PROJECT_NAME = kwargs.get('PROJECT_NAME', 'ncmMNIST')
        self.exp_name = kwargs.get('exp_name', self.PROJECT_NAME)

        self.PLOTS_PER_EPOCH = 1


        self.NOISE_DIM = kwargs.get('NOISE_DIM', 128)
        self.CONF_NOISE_DIM = kwargs.get('CONF_NOISE_DIM', 128)
        self.generator_decay=1e-6
        self.discriminator_decay=1e-6
        self.IMAGE_NOISE_DIM = kwargs.get('IMAGE_NOISE_DIM', 128)
        self.IMAGE_FILTERS = kwargs.get('IMAGE_FILTERS', [128, 64, 32])
        self.IMAGE_SIZE =  kwargs.get('IMAGE_SIZE', 32)
        self.ENCODED_DIM =  kwargs.get('ENCODED_DIM', 10)

        self.obs_state = kwargs.get('obs_state', 2)

        self.G_hid_dims = kwargs.get('G_hid_dims')  # in_d1  dn_out
        self.D_hid_dims = kwargs.get('D_hid_dims')  # 3x10x5x1


        self.CRITIC_ITERATIONS = kwargs.get('CRITIC_ITERATIONS', 5)
        self.LAMBDA_GP = kwargs.get('LAMBDA_GP', 0.1)  # It was 0.3

        self.learning_rate = kwargs.get('learning_rate', 2 * 1e-5)
        self.betas = (0.5, 0.9)
        self.Synthetic_Sample_Size = kwargs.get('Synthetic_Sample_Size', 20000)
        self.intv_Sample_Size = kwargs.get('intv_Sample_Size', 20000)
        self.ex_row_size = kwargs.get('ex_row_size', 20)
        self.batch_size = kwargs.get('batch_size', 100)  # from 256
        self.intv_batch_size = kwargs.get('intv_batch_size', 100)  # from 256
        self.num_epochs =  kwargs.get('num_epochs', 300)
        self.STOPAGE1 = 50
        self.STOPAGE2 = 20000
        self.lr_dec = 1

        self.curr_epoochs = 0
        self.curr_iter = 0

        # gumbel-softmax
        self.temp_min = kwargs.get('temp_min', 0.1)
        self.ANNEAL_RATE = 0.000003
        self.Temperature = kwargs.get('Temperature', 1)



        self.SAVE_MODEL = True
        self.LOAD_MODEL = False
        self.LOAD_TRAINED_CONTROLLER = False
        self.load_which_models={}

        # self.DEVICE = get_freer_gpu()
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        now = datetime.now()
        self.curDATE = now.strftime("%b_%d_%Y")
        self.curTIME = now.strftime("%H_%M")



        ret = set_truedag()
        self.DAG_desc, self.Complete_DAG_desc, self.Complete_DAG, self.complete_labels, self.Observed_DAG, self.label_names, self.label_dim, self.image_labels, self.rep_labels, self.interv_queries, self.latent_conf, \
            self.confTochild, self.exogenous, self.train_mech_dict,self.plot_title = ret


        self.cf_samples = self.Synthetic_Sample_Size
        self.num_labels = len(self.label_names)
        main_path= kwargs.get('main_path', f"./SAVED_EXPERIMENTS")
        # saving model and results
        self.new_experiment= kwargs.get('new_experiment', True)


        if self.new_experiment == True:
            os.makedirs(main_path ,exist_ok=True)
            # saved_path = main_path + self.Complete_DAG_desc + "/" + self.Exp_name+"/"+ self.curDATE + "-" + self.curTIME
            saved_path = main_path+ "/" + self.exp_name
            self.SAVED_PATH = kwargs.get('SAVED_PATH', saved_path)


            INSTANCES = {}
            INSTANCES["last_exp"] = self.SAVED_PATH
            with open(main_path +"/SHARED_INFO.txt", 'w') as fp:
                fp.write(json.dumps(INSTANCES))


    def anneal_temperature(self, tot_iters):

        # if (tot_iters) % 100 == 1:
        self.Temperature = np.maximum(
            self.Temperature * np.exp(-self.ANNEAL_RATE * tot_iters),
            self.temp_min)
        print(tot_iters, ":Temperature", self.Temperature)
