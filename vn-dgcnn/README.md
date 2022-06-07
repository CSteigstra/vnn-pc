# Vector Neurons++: Extending Neural Dimensionality and Generalizing Activation Functions for Vector Neuron Networks

This code base is forked from VN-PointNet <a href="https://github.com/FlyingGiraffe/vnn-pc/" target="_blank">Deng et al.</a>.
We introduce the addition of arbitrary activation and inclusion of normals in the VN-layers, specifically for PointNet.
For DGCNN we refer to https://github.com/CSteigstra/vnn-pc which is based on its correct implementation from https://github.com/FlyingGiraffe/vnn-pc .

## Overview
`vnn++` is the author's implementation of Vector Neuron Networks with PointNet and DGCNN backbones. The current version only supports PointNet for Modelnet40 classification.

## Environment
```
conda env create -f dl2_gpu.yml
conda activate dl2
# or 
source activate dl2
```

## Data Preparation

+ Classification normals: Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

## Usage

### Classification on ModelNet40
Training
```
# Author's (Deng et al.) LeakyReLU
python main_cls.py --exp_name SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot z
python main_cls.py --exp_name SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot z --normal
# Ours
python main_cls.py --exp_name SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot z  --activ leaky_relu
python main_cls.py --exp_name SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot z  --activ leaky_relu --normal
```

Evaluation
```
# Author's (Deng et al.) LeakyReLU
python main_cls.py --exp_name SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot so3
python main_cls.py --exp_name SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot so3 --normal
# Ours
python main_cls.py --exp_name=SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot so3 --activ leaky_relu
python main_cls.py --exp_name=SAVE_DIR --batch_size 12 --test_batch_size 8 --model eqcnn --rot so3 --activ leaky_relu --normal
```

## Citation
In the works. Refer to our github for now. ^.^

## License
MIT License

## Acknowledgement
The structure of this codebase is borrowed from this pytorch implementataion of [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [DGCNN](https://github.com/WangYueFt/dgcnn) as well as [this implementation](https://github.com/AnTao97/dgcnn.pytorch).
