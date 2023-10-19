import numpy as np
from torch.utils.data import Dataset

from ..utility.image import get_disk_mask


class SpotPatchDataset(Dataset):

    def __init__(self, embs, cnts, cnts_minmax, mask_size):
        super().__init__()
        self.labels = cnts.columns.to_list()
        cnts = cnts.to_numpy()
        self.x = embs
        self.y = cnts
        self.n_features = self.x.shape[-1]
        self.n_outcomes = self.y.shape[-1]
        self.cnts_minmax = cnts_minmax
        self.mask_size = mask_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def normalize_spotpatches(sps):
    cnts = sps['cnts']
    cnts = cnts.astype('float32')
    cmin, cmax = cnts.min(0).to_numpy(), cnts.max(0).to_numpy()
    cnts[:] = (cnts.to_numpy() - cmin) / (cmax - cmin + 1e-12)
    sps['cnts'] = cnts
    sps['cnts_minmax'] = np.stack([cmin, cmax])
    return sps


def get_spot_mask(radius, circular):
    if circular:
        mask = get_disk_mask(radius)
    else:
        size = int(np.ceil(radius)) * 2 + 1
        mask = np.ones((size, size), dtype=bool)
    return mask


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def make_dataset(slidebag):
    sps = slidebag.patchify()
    normalize_spotpatches(sps)
    dataset = SpotPatchDataset(**sps)
    return dataset
