import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from sketch2code.helpers import viz_grid


class CNNModule:
    
    def compute_act_layers(self, x: torch.tensor) -> Dict[str, torch.tensor]:
        return {}
    
    
def viz_first_conv_layer(conv2d, padding=1, plt=None, figsize: Tuple[int, int]=None, fontsize=20):
    """
    :params plt: pass pyplot if you want to display as well
    """
    weight = conv2d.weight
    min_weight = torch.min(weight)
    w = (weight - min_weight) / (torch.max(weight) - min_weight)
    w = w.permute((0, 2, 3, 1))
    w = viz_grid(w.detach().cpu().numpy(), padding)
    
    if plt is not None:
        fig = plt.figure(figsize=figsize)
        fig.suptitle("activation of first convolution layer", fontsize=fontsize)
        plt.imshow(w)
    return w
    
    
def viz_activation_layers(x: torch.tensor, model: CNNModule, padding: int=3, padding_color=1, plt=None, plt_config: dict=None):
    """
    :params plt: pass pyplot if you want to display as well
    :params plt_config: configuration for plt, which contains
                        `figsize`: size of the figure
                        `subplot_size`: (nrows, ncols) if you want to show more than one image in one figure, default is (1, 1)
                        `cmap`: default is interno
                        `fontsize`: default is 14
                        `interpolation`: default is nearest
    """
    acts = model.compute_act_layers(x)
    viz_acts = {
        k: viz_grid(v.cpu().numpy(), padding=padding, padding_color=padding_color)
        for k, v in acts.items()
    }
    
    if plt is not None:
        # displaying the activations
        keys = sorted(viz_acts.keys())
        plot_imgs(
            [{'title': f'{k} activations', 'data': viz_acts[k]} for k in keys],
            plt,
            figsize=plt_config['figsize'],
            subplot_size=plt_config.get('subplot_size', (1, 1)),
            interpolation=plt_config.get('interpolation', 'nearest'),
            cmap=plt_config.get('cmap', 'inferno'),
            fontsize=plt_config.get('fontsize', 14),
        )
    
    return viz_acts


def plot_imgs(images: List[dict], plt, figsize: Tuple[int, int], subplot_size: Tuple[int, int], interpolation="nearest", cmap="inferno", fontsize=14):
    """
    :params images: images to plot, a list of dictionaries, each contains image data (`data` attribute), and other metadata such as `title`.
    """
    n_imgs_per_fig = np.prod(subplot_size)
    for i in range(0, len(images), n_imgs_per_fig):
        # get images to display per figure
        fig_imgs = images[i:i+n_imgs_per_fig]
        
        fig = plt.figure(figsize=figsize)
        for j in range(len(fig_imgs)):
            ax = plt.subplot(subplot_size[0], subplot_size[1], j + 1)
            ax.imshow(fig_imgs[j]['data'], interpolation=interpolation, cmap=cmap)
            ax.set_title(fig_imgs[j]['title'], fontsize=fontsize)
