import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .io import save_image
from .image import get_disk_mask


def make_gif(arrays, filename, axis=0, duration=100):

    slices = [
            arrays.take(indices=i, axis=axis)
            for i in range(arrays.shape[axis])]

    frames = [Image.fromarray(s) for s in slices]
    frame_one = frames[0]
    frame_one.save(
            filename, format='GIF', append_images=frames,
            save_all=True, duration=duration, loop=0)
    print(filename)


def mat_to_img(
        x, white_background=True, transparent_background=False,
        cmap='turbo', minmax=None, verbose=False, dtype='uint8'):
    mask = np.isfinite(x)
    x = x.astype(np.float32)
    if minmax is None:
        minmax = (np.nanmin(x), np.nanmax(x) + 1e-12)
    if verbose:
        print('minmax:', minmax)
    x -= minmax[0]
    x /= minmax[1] - minmax[0]
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    x = cmap(x)
    if white_background:
        x[~mask] = 1.0
    if transparent_background:
        x[~mask, -1] = 0.0
    min_val = np.iinfo(dtype).min
    max_val = np.iinfo(dtype).max
    x = x * (max_val - min_val) + min_val
    x = x.astype(dtype)
    return x


def plot_matrix(x, outfile, **kwargs):
    img = mat_to_img(x, **kwargs)
    save_image(img, outfile)


def plot_spots(
        img, cnts, locs, radius, outfile, cmap='turbo',
        weight=0.8, disk_mask=True, standardize_img=False):
    cnts = cnts.astype(np.float32)

    img_dtype = img.dtype
    img_min_val = np.iinfo(img_dtype).min
    img_max_val = np.iinfo(img_dtype).max
    img = img.astype(np.float32)
    img = (img - img_min_val) / (img_max_val - img_min_val)

    if standardize_img:
        if np.isclose(0.0, np.nanstd(img, (0, 1))).all():
            img[:] = 1.0
        else:
            img -= np.nanmin(img)
            img /= np.nanmax(img) + 1e-12

    cnts -= np.nanmin(cnts)
    cnts /= np.nanmax(cnts) + 1e-12

    cmap = plt.get_cmap(cmap)
    if disk_mask:
        mask_patch = get_disk_mask(radius)
    else:
        mask_patch = np.ones((radius*2, radius*2)).astype(bool)
    indices_patch = np.stack(np.where(mask_patch), -1)
    indices_patch -= radius
    for ij, ct in zip(locs, cnts):
        color = np.array(cmap(ct)[:3])
        indices = indices_patch + ij
        img[indices[:, 0], indices[:, 1]] *= 1 - weight
        img[indices[:, 0], indices[:, 1]] += color * weight
    img = img * (img_max_val - img_min_val) + img_min_val
    img = img.astype(img_dtype)
    save_image(img, outfile)


def plot_history(history, prefix):
    plt.figure(figsize=(16, 16))
    groups = set([e.split('_')[-1] for e in history[0].keys()])
    groups = np.sort(list(groups))
    for i, grp in enumerate(groups):
        plt.subplot(len(groups), 1, 1+i)
        for metric in history[0].keys():
            if metric.endswith(grp):
                hist = np.array([e[metric] for e in history])
                hmin, hmax = hist.min(), hist.max()
                label = f'{metric} ({hmin:+013.6f}, {hmax:+013.6f})'
                hist -= hmin
                hist /= (hmax - hmin) + 1e-12
                plt.plot(hist, label=label)
        plt.legend()
        plt.ylim(0, 1)
        plt.xlim(0, len(hist))
    outfile = f'{prefix}history.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)
