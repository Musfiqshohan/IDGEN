from XrayLLM.Chexray_diffusion.cheff.ldm.inference import CheffLDMT2I
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor

# from cheff import CheffLDMT2I

device = 'cpu'
sdm_path = './XrayLLM/Chexray_diffusion/trained_models/cheff_diff_t2i.pt'
ae_path = './XrayLLM/Chexray_diffusion/trained_models/cheff_autoencoder.pt'

cheff_t2i = CheffLDMT2I(model_path=sdm_path, ae_path=ae_path, device=device)

# prompt = 'Large right-sided pleural effusion.'
prompt = 'Pneumothorax, pleural effusion.'


print(prompt)

img = cheff_t2i.sample(
    conditioning=prompt,
    sampling_steps=100,
    eta=1.0,
    decode=True
)

img.clamp_(-1, 1)
img = (img + 1) / 2
ret =to_pil_image(img[0])

x= ret.save("./XrayLLM/Chexray_diffusion/assets/xray.jpg")

