import numpy as np
from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.n_features = self.x.shape[-1]
        self.n_outcomes = self.y.shape[-1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_grid(*shape):
    ranges = [range(s) for s in shape]
    return np.stack(np.meshgrid(*ranges, indexing='ij'), -1)


def get_locs_3d(shape, z):
    locs_2d = get_grid(*shape)
    locs_3d = np.concatenate(
                [locs_2d, np.full_like(locs_2d[..., [0]], z)], -1)
    return locs_3d


def get_pixels_locs_3d(values, zlocs):
    shapes = [v.shape[:2] for v in values]  # (height, width)
    locs = [
            get_locs_3d(shape, z)
            for shape, z in zip(shapes, zlocs)]
    locs_flat = np.concatenate(
            [loc.reshape(-1, loc.shape[-1]) for loc in locs])
    values_flat = np.concatenate(
            [v.reshape(-1, v.shape[-1]) for v in values])
    return values_flat, locs_flat


def make_dataset(slidebag):
    values, locs = slidebag.flatten_spatial()
    dataset = SimpleDataset(x=locs, y=values)
    return dataset
