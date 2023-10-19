import os

import numpy as np
import torch

from ..utility.io import save_pickle, load_pickle
from ..data.spotpatch import make_dataset
from .nn import FeedForwardReduceNet, train_model


# TODO: refactor with GenExpGenerator3D
class GenExpPredictor:

    def __init__(self, n_states, model_path=None):

        self.model_path = model_path
        self.model_class = FeedForwardReduceNet
        self.n_states = n_states

    def train(self, slidebag, cache, **kwargs):

        dataset = make_dataset(slidebag)

        self.cnts_minmax = dataset.cnts_minmax
        self.mask_size = dataset.mask_size

        path = self.model_path
        model_exists = os.path.exists(path)
        if cache and model_exists:
            self.load_model(path)
        else:
            self._train(dataset=dataset, **kwargs)
        if cache and not model_exists:
            self.save_model(path)

    def _train(
            self, dataset, learning_rate, epochs, batch_size,
            device='cuda', **kwargs):

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
        for i in range(self.n_states):
            model, history, trainer = train_model(
                model_class=self.model_class,
                model_kwargs=model_kwargs,
                **kwargs)
            model.eval()
            model_list.append(model)
            history_list.append(history)
            trainer_list.append(trainer)

        self.model_list = model_list
        self.history_list = history_list
        self.trainer_list = trainer_list

    def save_model(self, path):
        save_pickle(self.model_list, path)

    def load_model(self, path):
        self.model_list = load_pickle(path)

    def predict_raw(self, x, indices=None):
        results_list = [
                _predict_raw_single(model, x, indices)
                for model in self.model_list]
        y_list = [e[0] for e in results_list]
        z_list = [e[1] for e in results_list]
        y = np.median(y_list, axis=0)
        z = np.median(z_list, axis=0)
        return y, z

    def predict(self, x, indices=None):
        y, z = self.predict_raw(x, indices)
        cnts_minmax = self.cnts_minmax[:, indices]
        y = y * (cnts_minmax[1] - cnts_minmax[0]) + cnts_minmax[0]
        y = y / self.mask_size

        return y, z


def _predict_raw_single(model, x, indices=None):
    x = torch.tensor(x, device=model.device)

    z = model.inp_to_lat(x)
    y = model.lat_to_out(z, indices=indices)

    y = y.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    return y, z
