import numpy as np
import torch
from sklearn.neighbors import RadiusNeighborsRegressor

from ..utility.io import save_pickle, load_pickle
from ..utility.visual import plot_history
from ..data.threedim import make_dataset
from .nn import FeedForwardNet, train_multistate


class GenExpModel:

    def __init__(self, model_class, n_states, model_path=None):

        self.model_path = model_path
        self.model_class = model_class
        self.n_states = n_states

    def train(self, slidebag, **kwargs):
        cache = kwargs.pop('cache')
        dataset = make_dataset(slidebag)
        self._train(dataset=dataset, **kwargs)
        if cache:
            self.save_model(self.model_path)
            self.plot_history(self.model_path.split('.')[0])

    def _train(self, **kwargs):
        out = train_multistate(
                model_class=self.model_class,
                n_states=self.n_states, **kwargs)
        model_list, history_list, trainer_list = out
        self.model_list = model_list
        self.history_list = history_list
        self.trainer_list = trainer_list

    def plot_history(self, path):
        for history in self.history_list:
            plot_history(history, path)

    def save_model(self, path):
        save_pickle(self.model_list, path)

    def load_model(self, path):
        self.model_list = load_pickle(path)


class GenExpGenerator3d(GenExpModel):

    def __init__(self, **kwargs):
        super().__init__(model_class=FeedForwardNet, **kwargs)

    def predict(self, x):
        y_list = [
                _predict(model, x)
                for model in self.model_list]
        y = np.median(y_list, axis=0)
        return y

    # def train(self, slidebag, **kwargs):
    #     dataset = make_dataset(slidebag)
    #     x, y = dataset.x, dataset.y
    #     model = get_neighbor_model(
    #             radius=0.2, weights='distance')
    #     model.fit(x, y)
    #     self.model = model

    # def predict(self, x):
    #     shape = x.shape[:-1]
    #     x = x.reshape((-1, x.shape[-1]))
    #     y = self.model.predict(x)
    #     y = y.reshape(shape+y.shape[-1:])
    #     return y


def _predict(model, x):
    x = torch.tensor(x, device=model.device)
    y = model.net(x)
    y = y.cpu().detach().numpy()
    return y


def get_neighbor_model(radius, weights):
    model = RadiusNeighborsRegressor(
            radius=radius, weights=weights)
    return model
