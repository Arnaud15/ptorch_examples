## Deep Learning examples in Pytorch
For now, we are getting started with ResNet re-implementations.


Check out a minimal demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fDUIkLjuU93LfLern0xHynnG4NWg-q7G?usp=sharing).


The current code in `models.py` is a re-implementation of the original ResNet [paper](https://arxiv.org/abs/1512.03385).


Next steps are implementing ideas from:
- The "Bag of tricks" [paper](https://arxiv.org/abs/1812.01187)
- The "revisiting ResNets" [paper](https://arxiv.org/abs/2103.07579)


Results on CIFAR-10's test set:
| Setup      | Accuracy |
| ----------- | ----------- |
| Baseline      | 0.915      |
| Baseline+ResNet simple tweaks   | TBD        |


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

Next up:
- More training epochs with / without MixUp
- Label smoothing
