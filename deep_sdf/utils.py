#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Comment out if not ssh'ing in
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import os, json
from typing import List, NoReturn

cmap = plt.get_cmap('viridis')

def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
        sdf = decoder(inputs)[:, :1]
    else:
        try:
            latent_repeat = latent_vector.expand(num_samples, -1)
            inputs = torch.cat([latent_repeat, queries], 1)
            with torch.no_grad():
                sdf = decoder(inputs)[:, :1]
        except:
            raise RuntimeError("Failed to decode SDF")

    return sdf


def decode_warping(decoder, latent_vector, queries, output_sdf=False):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    with torch.no_grad():
        warped, sdf = decoder(inputs, output_warped_points=True)

    if output_sdf:
        return warped, sdf
    else:
        return warped



def moving_average(x: np.array, window_size: int = 5) -> np.array:
    """
    Compute moving average of input array.

    :param x: np.array
    :param window_size: Size of window to take average upon
    """
    y = np.zeros(x.shape)
    y[:window_size] = x[:window_size]
    for i in range(window_size, len(x)):
        y[i] = np.mean(x[i - window_size:i])
    return y


def plot_all_iterations(log_dir: str, csv_name: str = 'train_iterations.csv', fname: str = 'training_plot.png',
                        init_epoch: int = 0, last_epoch: int = -1):

    df = pd.read_csv(os.path.join(log_dir, csv_name))
    if last_epoch == -1:
        init_epoch = df['epoch'].min()
        last_epoch = df['epoch'].max()
    df = df[df['epoch'] >= init_epoch]
    df = df[df['epoch'] <= last_epoch]

    errors = df.groupby('epoch').std().to_numpy().transpose(1, 0)
    data = df.groupby('epoch').mean().to_numpy().transpose(1, 0)
    epochs = range(int(init_epoch), int(last_epoch + 1))

    color = cmap(np.random.uniform(0, 1))
    plt.plot(epochs, data[1], c=color)[0]
    plt.fill_between(epochs, data[1] - errors[1], data[1] + errors[1],
                     alpha=0.2,
                     color=color,
                     linewidth=2)
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, fname))
    plt.close()



def plot_train_test(log_dirs: List[str], fname: str = 'summary.png', window_size: int = 5, title: str = '',
                    init_epoch: int = 0, last_epoch: int = -1) -> NoReturn:
    """
    Returns array of plots of training and samples data for each respective directories. Assumes that the data is kept in `train_test.csv`, as per the scripts in ``training/``, with column labels `loss` and `accuracy`.

    :param log_dirs: List of directories containing .csv
    :param fname: Name of output file
    :param window_size: Window size of moving average that is applied to ``col``
    :param title: Title of output plot
    :param init_epoch: Epoch to start plot from
    :param last_epoch: Final epoch to stop plotting
    """
    labels = []
    fig, axes = plt.subplots(len(log_dirs))
    if len(log_dirs) == 1:
        axes = [axes]
    for ax, log_dir in zip(axes, log_dirs):

        with open(join(log_dir, 'args.json'), 'r') as f:
            args = json.load(f)

        df = pd.read_csv(os.path.join(log_dir, 'train_test.csv'))

        if last_epoch == -1:
            init_epoch = df['epoch'].min()
            last_epoch = df['epoch'].max()

        df = df[df['epoch'] >= init_epoch]
        if last_epoch != -1:
            df = df[df['epoch'] < last_epoch]
        train_plot = ax.plot(moving_average(df['loss'], window_size=window_size), color='red', label='Train')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss (averaged)')
        plt.grid()

        ax2 = ax.twinx()
        test_plot = ax2.plot(moving_average(df['accuracy'], window_size=window_size), color='blue', label='Test')
        ax2.set_ylabel('Validation accuracy (averaged)')

        ax.legend(train_plot + test_plot, ['Train', 'Validation'], loc='best')
        plt.title('trained on {}'.format('dataset'))
    fig.suptitle(title)
    plt.savefig(os.path.join(log_dir, fname))
    plt.close()
