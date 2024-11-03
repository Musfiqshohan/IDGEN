import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional


def sample_gumbel(Exp, shape, eps=1e-20):
    U = torch.rand(shape).to(Exp.DEVICE)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(Exp, logits, temperature, gumbel_noise):

    if gumbel_noise==None:
        gumbel_noise= sample_gumbel(Exp, logits.size())

    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(Exp, logits, temperature, gumbel_noise=None, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    output_dim =logits.shape[1]
    y = gumbel_softmax_sample(Exp, logits, temperature, gumbel_noise)

    if not hard:
        ret = y.view(-1, output_dim)
        return ret

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    # ret = y_hard.view(-1, output_dim)
    ret = y_hard.view(-1, output_dim)
    return ret


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        # add the innermost block
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        # print(unet_block)

        # add intermediate block with nf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout)

        # gradually reduce the number of filters from nf * 8 to nf.
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        # add the outermost block
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)



class DigitImageGenerator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DigitImageGenerator, self).__init__()

        self.noise_dim = kwargs['noise_dim']
        self.parent_dims = kwargs['parent_dims']

        num_filters= kwargs['num_filters']
        self.output_dim = kwargs['output_dim']


        print(f'Image Generator init: noise dim: {self.noise_dim}, parent dim:{self.parent_dims}  outdim {self.output_dim}')


        # Hidden layers
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()



        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:

                if self.parent_dims == 0:
                    first_layer_dim = num_filters[i]
                else:
                    first_layer_dim = int(num_filters[i] / 2)


                # For input
                input_deconv = torch.nn.ConvTranspose2d(self.noise_dim, first_layer_dim, kernel_size=4, stride=1,padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Batch normalization
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(first_layer_dim))

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For label
                if self.parent_dims !=0:
                    label_deconv = torch.nn.ConvTranspose2d( self.parent_dims , first_layer_dim, kernel_size=4,stride=1, padding=0)
                    self.hidden_layer2.add_module('label_deconv', label_deconv)

                    # Batch normalization
                    self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(first_layer_dim))

                    # Activation
                    self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2,padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], self.output_dim, kernel_size=4, stride=2, padding=1)
        # out = torch.nn.ConvTranspose2d(num_filters[i], self.output_dim, kernel_size=3, stride=1, padding=1)  #if we want 32x32 for filter [256, 128, 64, 32]
        self.output_layer.add_module('out', out)

        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, noise, parent_labels):
        noises = torch.cat(noise, 1)
        h1 = self.hidden_layer1(noises)
        if len(parent_labels)!=0:
            parent_labels = torch.cat(parent_labels, 1)  # there will be always some parent to the digit image, otherwise gen_labels is just noise.
            # parent_labels=  parent_labels.view(-1, parent_labels.shape[1], 1, 1)
            h2 = self.hidden_layer2(parent_labels)
            x = torch.cat([h1, h2], 1)
        else:
            x=h1

        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out



# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class ClassificationNet(nn.Module):
    def __init__(self, **kwargs):
        super(ClassificationNet, self).__init__()
        self.noise_dim = kwargs['noise_dim']
        self.parent_dims = kwargs['parent_dims']
        self.output_dim = kwargs['output_dim']

        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, self.output_dim)

        self.conv1 = nn.Conv2d(self.noise_dim+self.parent_dims, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,self.output_dim)

    def forward(self, Exp,  x, **kwargs):
        x= torch.cat(x, 1)

        # x1 = self.conv1(x)
        # x2 =F.max_pool2d(x1, 2)
        # x = F.relu(x2)
        # # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)


        #-------
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return x

        output_feature = gumbel_softmax(Exp, x, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)

        return output_feature


class ConditionalClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(ConditionalClassifier, self).__init__()

        self.noise_dim = kwargs['noise_dim']
        self.parent_dims = kwargs['parent_dims']
        self.output_dim = kwargs['output_dim']

        self.conv1 = nn.Conv2d(self.noise_dim+self.parent_dims, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,self.output_dim)

    def forward(self, Exp, noise, parents,  **kwargs):  #condition or noise

        x = torch.cat(noise+parents, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output_feature = gumbel_softmax(Exp, x, Exp.Temperature).to(Exp.DEVICE)

        return output_feature





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

        # input_dim = 3 + kwargs['exos_dim'] + kwargs['conf_dim']  #exogenous and confounding noise
        input_dim = 3 + kwargs['exos_dim']   #exogenous and confounding noise
        output_dim = kwargs['output_dim']

        print("input_dim",input_dim)

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
        x = torch.cat([noises, x], 1)

        print('Xshape', x.shape)

        # x = torch.cat(x, 1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        output_feature = gumbel_softmax(Exp, x, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)
        return output_feature