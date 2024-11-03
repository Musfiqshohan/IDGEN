.PHONY : train
og_train:
	CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=gpu og_train.py
train:
	CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=gpu train.py
train_joint:
	CUDA_VISIBLE_DEVICES=1  torchrun --standalone --nproc_per_node=gpu train_joint.py --train_pkl ../retrain_trick/rt_data/retrain_trick_train.pkl --val_pkl ../retrain_trick/rt_data/retrain_trick_val.pkl --datakey X,Y --condkey W1 --labkey W2 
train_uncond:
	CUDA_VISIBLE_DEVICES=4  python -im torch.distributed.launch --use_env --master_port=12356 --standalone --nproc_per_node=gpu train_uncond.py --train_pkl ../retrain_trick/rt_data/retrain_trick_train.pkl --datakey W1 --moddir w1_uncond_model --samdir w1_uncond_samples
trainimg:
	CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=gpu train_imgcond.py --train_pkl ../retrain_trick/rt_data/retrain_trick_train.pkl --val_pkl ../retrain_trick/rt_data/retrain_trick_val.pkl
.PHONY : sample
samplepict:
	CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True --select quadratic --genbatch 80 --w 0.5
.PHONY : samplenpz
samplenpz:
	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu sample.py --fid True
.PHONY : clean
clean:
	rm -rf __pycache__
	rm -rf model/*
	rm -rf sample/*

.PHONY : base
base_host: 
		CUDA_VISIBLE_DEVICES=0,1,2,4 torchrun --standalone --nproc_per_node=gpu base_host.py
