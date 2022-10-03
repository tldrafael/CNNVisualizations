# README 

See the repo tutorial on ![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15jDSPdGYmXplrqJUVZq_fPqyTbcZgXFk?usp=sharing).

The topics approached on this repo are:
  + Region importance at any layer using [Grad-CAM](https://arxiv.org/abs/1610.02391).
    + With or without guided-backpropagation.
  + Optimize an input to maximize a neuron or an output with gradient ascent.
    + Deep dream, optimization starting from an existing image.
  + Generate adversarial examples.

The results are presented for classification and semantic segmentation.

## Semantic Segmentation 

The images bellow illustrate the RTK Dataset.

### GradCAM

![](./cache/RTK_GradCAM.png)

### Input Synthesis

![](./cache/RTK_optimuminputs.png)

### Adversarial Examples

![](./cache/RTK_adversarials.png)

## Classification

The images bellow illustrate the ImageNet.

### GradCAM

![](./cache/imageNet_GradCAM.png)

### Input Synthesis

![](./cache/imageNet_optimuminputs.png)

### Adversarial Examples

![](./cache/imageNet_adversarials.png)

## References

+ https://github.com/jacobgil/pytorch-grad-cam/
+ https://aman.ai/cs231n/visualization/
+ http://cs231n.stanford.edu/slides/2021/lecture_14.pdf
+ https://distill.pub/2017/feature-visualization/
