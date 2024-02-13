import torch
from torch import nn
from torchvision.models import resnet18

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb



class MNISTEmbedding(nn.Module):
    def __init__(self, channels:int, dim:int, hw:int=32):
        super().__init__()
        # Simple MNIST CNN architecture
        out_shape = {28: 32 * 7 * 7,
                     32: 32 * 8 * 8}[hw]
        self.condEmbedding = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(out_shape, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb


class CXRayEmbedding(nn.Module):
    def __init__(self, dim:int):

        super().__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)



class JointEmbedding(nn.Module):
    """ Embedding for both discrete and image embeddings (jointly) """

    def __init__(self, num_labels:int, d_model:int, channels:int, dim:int, hw:int=32):
        super().__init__()
        self.img_embedder = MNISTEmbedding(channels, dim, hw)
        self.cond_embedder = ConditionalEmbedding(num_labels, d_model, dim)

        self.joiner = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, img, lab, drop_label=False, drop_image=False, threshold=0.1):
        bsz = img.shape[0]
        img_emb = self.img_embedder(img)
        lab_emb = self.cond_embedder(lab)
        if drop_label or drop_image:
            rand_drop = (torch.rand(bsz) < threshold).to(img.device)
        if drop_label:
            lab_emb[rand_drop] = 0
        if drop_image:
            img_emb[rand_drop] = 0

        return self.joiner(torch.cat([img_emb, lab_emb], dim=1))


class JointEmbedding2(nn.Module):
    """ Embedding for both discrete and image embeddings (jointly) """

    def __init__(self, num_labels_0:int, num_labels_1:int, d_model:int, channels:int, dim:int, hw:int=32):
        super().__init__()
        self.img_embedder = MNISTEmbedding(channels, dim, hw)
        self.cond_embedder_0 = ConditionalEmbedding(num_labels_0, d_model, dim)
        self.cond_embedder_1 = ConditionalEmbedding(num_labels_1, d_model, dim)

        self.joiner = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, img, lab_0, lab_1, threshold=0.1, drop_label=False, drop_image=False):
        bsz = img.shape[0]
        img_emb = self.img_embedder(img)
        lab_emb_0 = self.cond_embedder_0(lab_0)
        lab_emb_1 = self.cond_embedder_1(lab_1)

        if drop_label or drop_image:
            rand_drop = (torch.rand(bsz) < threshold).to(img.device)
        if drop_label:
            lab_emb_0[rand_drop] = 0
            lab_emb_1[rand_drop] = 0
        if drop_image:
            img_emb[rand_drop] = 0

        return self.joiner(torch.cat([img_emb, lab_emb_0, lab_emb_1], dim=1))        
        
