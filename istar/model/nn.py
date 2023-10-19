import os
from copy import deepcopy
import math

from ..utility.io import save_pickle, load_pickle
from ..utility.visual import plot_history

import numpy as np
import torch
from torch import nn
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def torch_corr(x, y):
    return torch.corrcoef(torch.stack([x, y]))[0, 1]


def train_model(
        dataset, batch_size, epochs,
        model=None, model_class=None, model_kwargs={},
        device='cuda', prefix=None, ckpt_path=None):

    if model is None:
        model = model_class(**model_kwargs)
    dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True)
    tracker_metric = MetricTracker()
    tracker_lr = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
            dirpath=prefix+'checkpoints/', save_last=True,
            every_n_epochs=10)
    cc_sd_path = prefix+'checkpoints/state.pickle'
    if os.path.exists(cc_sd_path):
        checkpoint_callback.load_state_dict(load_pickle(cc_sd_path))
    print('last model path:', checkpoint_callback.last_model_path)
    device_accelerator_dict = {
            'cuda': 'gpu',
            'cpu': 'cpu'}
    accelerator = device_accelerator_dict[device]
    trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[tracker_lr, tracker_metric, checkpoint_callback],
            deterministic='warn',
            accelerator=accelerator,
            devices=1,
            logger=True,
            enable_checkpointing=True,
            enable_progress_bar=True,
            default_root_dir=prefix)
    model.train()
    trainer.fit(
            model=model, train_dataloaders=dataloader, ckpt_path=ckpt_path)
    save_pickle(
            checkpoint_callback.state_dict(),
            cc_sd_path)
    tracker_metric.clean()
    history = tracker_metric.collection
    return model, history, trainer


def get_model(
        model_class, model_kwargs, dataset, prefix,
        epochs=None, device='cuda', load_saved=True, **kwargs):

    # load history if exists
    history_file = prefix + 'history.pickle'
    load_history = load_saved and os.path.exists(history_file)
    if load_history:
        history = load_pickle(history_file)
    else:
        history = []

    model = None
    checkpoint_file = 'last'

    # train model
    if (epochs is not None) and (epochs > 0):
        model, hist, trainer = train_model(
            model=model,
            model_class=model_class, model_kwargs=model_kwargs,
            dataset=dataset, epochs=epochs, device=device,
            prefix=prefix, ckpt_path=checkpoint_file, **kwargs)
        # trainer.save_checkpoint(checkpoint_file)
        # print(f'Model saved to {checkpoint_file}')
        history += hist
        save_pickle(history, history_file)
        print(f'History saved to {history_file}')
        plot_history(history, prefix)

    return model, history, trainer


def train_multistate(
        model_class, dataset, n_states, learning_rate, epochs, batch_size,
        device='cuda', prefix=None, **kwargs):

    if batch_size == 'auto':
        batch_size = min(128, len(dataset)//16)
    model_kwargs = {
            'n_inp': dataset.n_features,
            'n_out': dataset.n_outcomes,
            'lr': learning_rate,
            }
    kwargs['dataset'] = dataset
    kwargs['batch_size'] = batch_size
    kwargs['epochs'] = epochs
    kwargs['device'] = device

    model_list, history_list, trainer_list = [], [], []
    for i in range(n_states):
        model, history, trainer = get_model(
            model_class=model_class,
            model_kwargs=model_kwargs,
            prefix=prefix,
            **kwargs)
        model.eval()
        model_list.append(model)
        history_list.append(history)
        trainer_list.append(trainer)

    return model_list, history_list, trainer_list


class MetricTracker(pl.Callback):

    def __init__(self):
        self.collection = []

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        metrics = deepcopy(trainer.logged_metrics)
        self.collection.append(metrics)

    def clean(self):
        keys = [set(e.keys()) for e in self.collection]
        keys = set().union(*keys)
        for elem in self.collection:
            for ke in keys:
                if ke in elem.keys():
                    if isinstance(elem[ke], torch.Tensor):
                        elem[ke] = elem[ke].item()
                else:
                    elem[ke] = float('nan')


class FeedForwardNet(pl.LightningModule):

    def __init__(self, lr, n_inp, n_out, activation=None):
        super().__init__()
        self.lr = lr
        if activation is None:
            activation = nn.LeakyReLU(0.01, inplace=True)
        width = 1024
        dim = 512
        self.net = nn.Sequential(
                SinusoidalPositionEmbeddings(dim=dim),
                FeedForward(n_inp*dim, width, activation=activation),
                FeedForward(width, width, activation=activation),
                FeedForward(width, width, activation=activation),
                FeedForward(width, width, activation=activation),
                FeedForward(width, n_out),
                )
        self.save_hyperparameters()

    def forward(self, x, indices=None):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        l1 = l1_loss(y_pred, y)
        self.log('l1', l1, prog_bar=True)
        corr = torch_corr(y.flatten(), y_pred.flatten())
        self.log('1mcor', 1-corr, prog_bar=True)
        loss = l1
        self.log('loss', loss, prog_bar=False)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', np.log10(lr), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=10,
                threshold=0.0001, threshold_mode='rel',
                cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'loss',
                    'interval': 'epoch',
                    'frequency': 1
                    }
                }


class FeedForwardReduceNet(pl.LightningModule):

    def __init__(self, lr, n_inp, n_out):
        super().__init__()
        self.lr = lr
        self.net_lat = nn.Sequential(
                FeedForward(n_inp, 256),
                FeedForward(256, 256),
                FeedForward(256, 256),
                FeedForward(256, 256))
        self.net_out = FeedForward(
                256, n_out,
                activation=ELU(alpha=0.01, beta=0.01))
        self.save_hyperparameters()

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        y_pred = self.forward(x)
        y_mean_pred = y_pred.mean(-2)
        # TODO: try l1 loss
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('rmse', mse**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            # TODO: change activation to LeakyRelu(0.01)
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=512, wavelength_max=10000):
        super().__init__()
        self.dim = dim
        self.wavelength_max = wavelength_max  # multiples of 2pi

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(self.wavelength_max) / (half_dim - 1)
        embeddings = torch.exp(
                torch.arange(half_dim, device=device)
                * (-1) * embeddings)
        embeddings = t[..., None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.view(*embeddings.shape[:-2], -1)
        return embeddings
