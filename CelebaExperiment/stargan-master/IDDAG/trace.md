1. Generate images from StarGAN with [evaluate_pretrained.ipynb](evaluate_pretrained.ipynb).
   2. Original 128 image
   3. Original 256 image
   4. Stargan generated 128 images
5. Run [run_pretrained_model.py](..%2F..%2FEGSDE-master%2FIDDAG%2Frun_pretrained_model.py) in EGSDE on the 'Original 256'
images.
6. Run [compare_models.ipynb](compare_models.ipynb) to compare images between stargan and EGSDE.
7. Run [compare_models.ipynb](compare_models.ipynb) to estimate label distribution between stargan and EGSDE.
8. Run [conditional_generation.ipynb](conditional_generation.ipynb) to estimate label distribution of P(A|Young, do(Male=0))