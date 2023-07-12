import datetime
import logging
import os
import numpy as np
import pandas as pd
import subprocess
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from argparse import Namespace
from math import ceil
from typing import List, Type
from torch import nn


def _get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])\
            .decode('ascii').strip()
    except:
        return 'no_git'


def _get_git_revision_short_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])\
            .decode('ascii').strip()
    except:
        return 'no_git'


def _generate_output_dirname(prefix: str) -> str:
    git_hash = _get_git_revision_short_hash()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dirname = git_hash + '_' + now
    if prefix is not None:
        dirname = prefix + '_' + dirname
    return dirname


def generate_output_dirpath(module_path: str, output_rootname: str='output', prefix: str=None):
    module_dirpath = os.path.dirname(os.path.abspath(module_path))
    output_dirname = _generate_output_dirname(prefix)
    output_dirpath = f'{module_dirpath}/{output_rootname}/{output_dirname}'
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirpath)
    return output_dirpath


def generate_model_experiment_name(args: Namespace, is_test_mode: bool=False) -> str:
    return "_".join([
        args.model,
        args.pretrained if args.pretrained else 'scratch',
        f"ub{args.unfreeze_blocks}",
        'discretized' if args.discretized_image else 'original',
        args.optimizer,
        f"lr{args.lr}",
        f"wd{args.weight_decay}",
        f"ep{args.num_train_epochs}",
        f"bs{args.train_batch_size}",
        'test' if is_test_mode else 'train'
    ])


def generate_classifier_experiment_name(args: Namespace, is_test_mode: bool=False) -> str:
    return "_".join([
        args.model,
        args.pretrained,
        'discretized' if args.discretized_image else 'original',
        'test' if is_test_mode else 'train'
    ])


def get_logger(filename: str=None) -> logging.Logger:
    log_format = '%(asctime)s [%(levelname)-5.5s] %(message)s'
    log_formatter = logging.Formatter(log_format)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if filename is not None and filename != '':
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    return logger


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(model: Type[nn.Module], inputs):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = model(inputs).cpu()
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model: Type[nn.Module], images: Type[torch.tensor], 
                       labels: Type[torch.tensor], classes: List[str]):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    n_images = min(40, len(images))
    n_cols = min(4, n_images)
    n_rows = int(ceil(float(n_images) / n_cols))

    preds, probs = images_to_probs(model, images)
    images = images.cpu()
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 36))
    for idx in np.arange(n_images):
        ax = fig.add_subplot(n_rows, n_cols, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red")
        )
    return fig


def combine_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df_concat = pd.concat(dfs)
    group_df = df_concat.groupby(level=0)

    df_means = group_df.mean().applymap('{:.4f}'.format)
    df_stds = group_df.std().applymap('{:.4f}'.format)
    df = df_means.combine(df_stds, lambda x1, x2: x1 + ' ± ' + x2)
    return df
