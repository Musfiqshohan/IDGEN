
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from Classifier.lightningmodules.classification import Classification

import numpy as np
import torch
from pytorch_lightning import Trainer

################################ Classification
from dataclasses import dataclass
import os, os.path as osp
from typing import Any, ClassVar, Dict, List, Optional
from matplotlib import pyplot as plt


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb parameters
    wandb_project : str  = "classif_celeba"
    wandb_entity  : str  = "rahman89"       # name of the project
    save_dir      : str  = osp.join(os.getcwd())                    # directory to save wandb outputs
    weights_path  : str  = osp.join(os.getcwd(), "weights")

    # train or predict
    train : bool = True
    predict: bool = False

    gpu: int = 1
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    val_check_interval: float = 0.5

@dataclass
class TrainParams:
    """Parameters to use for the model"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40
    lr : int = 0.00001

@dataclass
class DatasetParams:
    """Parameters to use for the model"""
    # datamodule
    num_workers       : int         = 2         # number of workers for dataloadersint
    # root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets")   # '/kaggle/working'
    # root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets", "inputs")   # '/kaggle/working'
    root_dataset      : Optional[str] = "/local/scratch/a/rahman89/Datasets/celeba/"
    # root_dataset      : Optional[str] = "/local/scratch/a/rahman89/CelebAMask-HQ"
    batch_size        : int         = 1        # batch_size
    input_size        : tuple       = (224, 224)   # image_size

@dataclass
class InferenceParams:
    """Parameters to use for the inference"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40
    # ckpt_path: Optional[str] = osp.join(os.getcwd(), "weights", "ViTsmall.ckpt")
    ckpt_path: Optional[str] = osp.join("/local/scratch/a/rahman89/PycharmProjects/STAR_GAN/Classifier/",  "ViTsmall.ckpt")
    output_root :  str = osp.join(os.getcwd(), "output")
    lr: int = 0.00001



@dataclass
class SVMParams:
    """Parameters to edit for SVM training"""
    json_file           : str       = "outputs_stylegan/stylegan3/scores_stylegan3.json"
    np_file             : str       = "outputs_stylegan/stylegan3/z.npy"
    output_dir          : str       = "trained_boundaries_z_sg3"
    latent_space_dim    : int       = 512
    equilibrate         : bool      = False

@dataclass
class Parameters:
    """base options."""

    hparams       : Hparams         = Hparams()
    data_param    : DatasetParams   = DatasetParams()
    train_param   : TrainParams     = TrainParams()
    inference_param : InferenceParams = InferenceParams()
    svm_params      : SVMParams = SVMParams()

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance



def generatedCorrectly(model, image, trainer, att_val):

    image = image[0]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean, std),
        ]
    )

    data_list = []
    lbl = torch.zeros(40, 1)
    data_list.append([transform(image), lbl])


    predict_loader = DataLoader(dataset=data_list, batch_size=1, shuffle=False)
    prediction = trainer.predict(model, predict_loader)  # without fine-tuning
    for idx, data_input in enumerate(prediction):
        pred= data_input[2][0]


    att1, att2= att_val.keys()
    a1,a2= att_val.values()
    b1,b2=0,0
    if att1 in pred:
        b1 = 1
    if att2 in pred:
        b2 = 1

    if (a1,a2) == (b1,b2):
        # print(True)
        return True

    return False





################################# Generation
def sample_codes(generator, num, latent_space_type='Z', seed=0):
  """Samples latent codes randomly."""
  np.random.seed(seed)
  codes = generator.easy_sample(num)
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
    codes = generator.get_value(generator.model.mapping(codes))
  return codes

def generate_stylegan_images(generator, boundaries,  ATTRS, att_val, num_samples ):
    # @title { display-mode: "form", run: "auto" }

    # num_samples= 5  # @param {type:"slider", min:1, max:8, step:1}
    latent_space_type = 'Z'  # @param ['Z', 'W']
    synthesis_kwargs = {}

    # Female young: -1,-1
    # Female old: -3, 4
    # Male old: 3, 3
    # Male young: 3, -1
    # gender = 3  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    # age = -1  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}

    # for iter in range(5):
    noise_seed = torch.randint(0, 10000, (1,))  # @param {type:"slider", min:0, max:1000, step:1}

    latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
    # images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']   #generating base images

    new_codes = latent_codes.copy()
    for i, attr_name in enumerate(ATTRS):
        # new_codes += boundaries[attr_name] * eval(attr_name)
        new_codes += boundaries[attr_name] * att_val[attr_name]

    new_images = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
    # imshow(new_images, col=num_samples)

    return new_images





def plot_image_ara(img_ara, folder=None, title=None):
    rows=img_ara.shape[0]
    cols=img_ara.shape[1]

    print(rows,cols)

    f, axarr = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for c in range(cols):

        for r in range(rows):
            axarr[r, c].get_xaxis().set_ticks([])
            axarr[r, c].get_yaxis().set_ticks([])

            img= img_ara[r][c]
            # img= np.transpose(img, (1,2,0))
            axarr[r, c].imshow(img)


        f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if folder==None:
        plt.show()
    else:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{title}.png', bbox_inches='tight')

    plt.close()



def get_classifier(attributes, IMAGE_SIZE=128):


    #### load classifier
    config = Parameters()


    attr_dict= attributes

    checkpoint = torch.load(config.inference_param.ckpt_path)
    model = Classification(config.inference_param, attr_dict)
    model.load_state_dict(checkpoint["state_dict"])
    print('Classifier loaded')
    trainer = Trainer(devices=config.hparams.gpu, limit_train_batches=0, limit_val_batches=0)


    return model,trainer



from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Classifier.pre_trained import get_classifier
if __name__ == '__main__':


    label_path = "/local/scratch/a/rahman89/Datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
    attributes = open(label_path).readlines()[1].split(' ')
    attributes[-1] = attributes[-1].strip('\n')


    trainer = get_classifier(attributes)