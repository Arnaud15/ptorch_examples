## Deep Learning examples in Pytorch

### SimCLR
Simple Pytorch implementation in `src/simclr/simclr.py`.
An example notebook is provided where we look at a simplistic contrastive learning task, distinguishing points on the unit circle (available [here](https://github.com/Arnaud15/ptorch_examples/blob/master/simclr_unit_circle.ipynb) and in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VuH-5GxcCOoqPyBbMOkRwC1Zz39Kqz1W?usp=sharing)) to illustrate my recent [blog post about SimCLR](https://arnaudautef.com/deep%20learning/computer%20vision/contrastive%20learning/2021/11/08/simclr.html).

The original paper is available [here](https://arxiv.org/abs/2002.05709).

---

### ResNet re-implementations
The code in `src/models/resnet.py` is a re-implementation of the original ResNet [paper](https://arxiv.org/abs/1512.03385).


Next steps are implementing ideas from:
- The "Bag of tricks" [paper](https://arxiv.org/abs/1812.01187)
- The "revisiting ResNets" [paper](https://arxiv.org/abs/2103.07579)


Results on CIFAR-10's test set:
| Setup      | Accuracy |
| ----------- | ----------- |
| Baseline      | 0.915      |
| Baseline+ResNet simple tweaks   | 0.915        |


Baseline:
- 200 epochs
- learning rate 0.1
- cosine decay
- linear warmup of 5 epochs
- batch size 128
- weight decay 1e-4

ResNet simple tweaks:
- better downsampling, instead of stride 2 1x1 convolutions
- 3 convolutions 3x3 instead of a single 7x7 convolution in the stem

TODOs:
- More training epochs with / without MixUp
- Label smoothing
- Reduce weight decay
