Following instructions in this repo:
https://github.com/CausalAILab/NeuralCausalAbstractions

We ran the GAN-RNCM, run the following command
python3 -m Baselines.NeuralCausalAbstractionsMain.src.Napkin.napkin_main napkin sampling napkin_mnist gan --h-layers 3 --h-size 2 --scale-h-size --scale-u-size --batch-norm --gan-mode wgan --gan-arch biggan --repr auto_enc_conditional --rep-size 64 --rep-image-only --rep-h-layers 3 --rep-h-size 128 --img-size 32 --gpu 0

1. We trained the encoders with 300 epochs. and appeared visually consistent.
2. We trained the NCM with 1000 epochs.





