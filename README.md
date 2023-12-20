# CIFAR10 Benchmark in PyTorch
Easy-to-run model benchmark on CIFAR10.

With this repository you can:
- train VGG[1], ResNet[2]
- manage training conditions using OmegaConf
- plot the results on tensorboard
- build environment using docker

So far you cannot:
- train models on ImageNet
- train models using multiple GPUs
- load tensorflow weights

### What's New
- <b>Updated docker and config environments (Dec. 2023)</b>
- <b>EfficientNet models are no longer supported (Dec. 2023)</b>

## Benchmark Results

RandomCrop and LRFlip are used for data augmentation.

### From Scratch
Input size is 32x32.
<table><tbody>
<tr><th>VGG16: 93.3 %</th><th>ResNet18: 94.3%</th></tr>
<tr><th><img src="data/vgg16_test_acc.png" height="160"\> </th><th>
<img src="data/resnet18_test_acc.png" height="160"\></th></tr>
</table></tbody>

## Getting Started

### Prerequisites:
- Python 3.6+
- PyTorch 1.0+
- (optional) tensorboardX

### Docker

Install `docker-compose` and `nvidia-container-runtime` beforehand.

```bash
$ docker-compose build --build-arg UID="`id -u`" dev
$ docker-compose run dev
```

## Training on CIFAR10

You can select a model to train by specifying a config file.
```bash
$ python train.py --help
usage: train.py [-h] [--config CONFIG] [--tfboard TFBOARD]
                [--checkpoint_dir CHECKPOINT_DIR] [--resume RESUME]

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  path to config file
  --tfboard TFBOARD  tensorboard path for logging
  --checkpoint_dir CHECKPOINT_DIR
            directory where checkpoint files are saved
  --resume RESUME       checkpoint file path
```

example:

```bash
$ python train.py --config configs/vgg16.yaml --tfboard out
```

## TODOs

(TBD)

## References
[1] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" [paper](https://arxiv.org/abs/1409.1556) <br>
[2] K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition" [paper](https://arxiv.org/abs/1512.03385)<br>
[3] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" [paper](https://arxiv.org/abs/1905.11946)<br>
[4] [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)<br>
