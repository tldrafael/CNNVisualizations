import os
import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from IPython import display
import torch
import torchvision.transforms as T
import torchvision as tv
from torch.distributions.bernoulli import Bernoulli


augs = SimpleNamespace()
augs.normalize = T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
augs.normalize_invert = T.Normalize(mean=[-.485 / .229, -.456 / .224, -.406 / .225], std=[1 / .229, 1 / .224, 1 / .225])

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'imagenet_class_index.json'), 'r') as f:
    imagenet_json = json.load(f)
imagenet_label = [imagenet_json[str(k)][1] for k in range(len(imagenet_json))]


def get_image_from_input_tensor(inp_image, ix=0):
    return augs.normalize_invert(inp_image)[ix].permute(1, 2, 0).detach().cpu().numpy()


def get_input_tensor_from_image(image):
    assert len(image.shape) == 3, 'Image should have 3 axis'
    return augs.normalize(torch.Tensor(image).permute(2, 0, 1)[None])


def get_tensor_deciles(x, n_round=0, intervals=torch.linspace(0, 1, 11)):
    torch_quants = torch.quantile(x, q=intervals.to(x.device))
    return torch_quants.detach().cpu().numpy().round(n_round)


def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def normalize_minmax_gentle(x, upper=0.98, lower=0.02):
    x = (x - x.quantile(upper)) / (x.quantile(upper) - x.quantile(lower))
    return x.clip(min=0, max=1)


def plot_grid(grid, use_display=False, is_grid=False, nrows=1, ncols=None, figsize=None):
    if ncols is None:
        if isinstance(grid, list):
            ncols = len(grid)
        else:
            ncols = grid.shape[0]

    if isinstance(grid, np.ndarray):
        grid = torch.Tensor(grid)

    if not is_grid:
        grid = tv.utils.make_grid(grid, nrow=ncols)

    if figsize is None:
        plt.figure(figsize=(4 * ncols, 5 * nrows))
    else:
        plt.figure(figsize=figsize)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    if use_display:
        display.display(plt.gcf())


def decide_randomly(p, thrs=.5):
    return Bernoulli(p).sample((1,)).item() > thrs
