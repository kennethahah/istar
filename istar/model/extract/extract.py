import os

import numpy as np
from einops import rearrange, reduce, repeat
import torch

from ...utility.image import smoothen
from .hipt.hipt_model_utils import eval_transforms
from .hipt.hipt_4k import HIPT_4K


def smoothen_embs(
        embs, size, kernel,
        method='cv', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [
                        smoothen(
                            c[..., np.newaxis], size=size,
                            kernel=kernel, backend=method,
                            device=device)[..., 0]
                        for c in em]
            else:
                smoothened = smoothen(em, size, method, device=device)
        else:
            smoothened = em
        out[grp] = smoothened
    return out


def smoothen_embeddings(embs, device):
    embs = smoothen_embs(
            embs, size=16, kernel='uniform', groups=['cls'],
            device=device)
    embs = smoothen_embs(
            embs, size=4, kernel='uniform', groups=['sub'],
            device=device)
    return embs


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    x = np.pad(
            x,
            (
                (0, shape_ext[0] - x.shape[0]),
                (0, shape_ext[1] - x.shape[1]),
                (0, 0)),
            mode='edge')
    tiles_shape = np.array(x.shape[:2]) // patch_size
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> h1 w1 h w c',
    #         h=patch_size, w=patch_size)
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> (h1 w1) h w c',
    #         h=patch_size, w=patch_size)
    tiles = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size  # TODO: change to patch_size[0]
        b0 = a0 + patch_size  # TODO: change to patch_size[0]
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size  # TODO: change to patch_size[1]
            b1 = a1 + patch_size  # TODO: change to patch_size[1]
            tiles.append(x[a0:b0, a1:b1])

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    return tiles, shapes


def get_embeddings_sub(model, x):
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def get_embeddings_cls(model, x):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        __, x_sub4k = model.forward_all4k(x[None])
    x_sub4k = x_sub4k.cpu().detach().numpy()
    x_sub4k = x_sub4k[0].transpose(1, 2, 0)
    return x_sub4k


def get_embeddings(img, model, pretrained_path=None, device='cuda'):
    '''
    Extract embeddings from histology tiles
    Args:
        tiles: Histology image tiles.
            Shape: (N, H, W, C).
            `H` and `W` are both divisible by 256.
            Channels `C` include R, G, B, foreground mask.
    Returns:
        emb_cls: Embeddings of (256 x 256)-sized patches
            Shape: (H/256, W/256, 384)
        emb_sub: Embeddings of (16 x 16)-sized patches
            Shape: (H/16, W/16, 384)
    '''

    tile_size = 4096
    tiles, shapes = patchify(img, patch_size=tile_size)

    patch_size = (256, 256)
    subpatch_size = (16, 16)
    n_subpatches = tuple(
            a // b for a, b in zip(patch_size, subpatch_size))

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        x_mid, x_sub = get_embeddings_sub(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)
    del tiles
    torch.cuda.empty_cache()
    emb_mid = rearrange(
            emb_mid, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=shapes['tiles'][0], w1=shapes['tiles'][1])

    emb_cls = get_embeddings_cls(model, emb_mid)
    del emb_mid, model
    torch.cuda.empty_cache()

    shape_orig = np.array(shapes['original']) // subpatch_size

    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
                np.array([e[..., i] for e in emb_sub]),
                '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
                h1=shapes['tiles'][0], w1=shapes['tiles'][1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_sub.append(chan)
    del emb_sub

    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
                np.array([e[..., i] for e in emb_cls]),
                'h12 w12 -> (h12 h3) (w12 w3)',
                h3=n_subpatches[0], w3=n_subpatches[1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_cls.append(chan)
    del emb_cls

    return chans_cls, chans_sub


def get_embeddings_shift(
        img, model, margin=256, stride=64,
        pretrained_path=None, device='cuda'):
    # margin: margin for shifting. Divisble by 256
    # stride: stride for shifting. Divides `margin`.
    factor = 16  # scaling factor between cls and sub. Fixed
    shape_emb = np.array(img.shape[:2]) // factor
    chans_cls = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(192)]
    chans_sub = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(384)]
    start_list = list(range(0, margin, stride))
    n_reps = 0
    for start0 in start_list:
        for start1 in start_list:
            print(f'shift {start0}/{margin}, {start1}/{margin}')
            stop0, stop1 = -margin+start0, -margin+start1
            im = img[start0:stop0, start1:stop1]
            cls, sub = get_embeddings(
                    im, model, pretrained_path=pretrained_path, device=device)
            del im
            sta0, sta1 = start0 // factor, start1 // factor
            sto0, sto1 = stop0 // factor, stop1 // factor
            for i in range(len(chans_cls)):
                chans_cls[i][sta0:sto0, sta1:sto1] += cls[i]
            del cls
            for i in range(len(chans_sub)):
                chans_sub[i][sta0:sto0, sta1:sto1] += sub[i]
            del sub
            n_reps += 1

    mar = margin // factor
    for chan in chans_cls:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0
    for chan in chans_sub:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0

    return chans_cls, chans_sub


def get_hipt_model(path, device):
    model256_path, model4k_path = None, None
    if path is not None:
        print(f'Model loaded from {path}')
        model256_path = os.path.join(path, 'vit256_small_dino.pth')
        model4k_path = os.path.join(path, 'vit4k_xs_dino.pth')
    model = HIPT_4K(
            model256_path=model256_path,
            model4k_path=model4k_path,
            device256=device, device4k=device)
    model.eval()
    return model


def extract_raw(
        image, shift=True, pretrained_path=None,
        device='cuda'):

    model = get_hipt_model(path=pretrained_path, device=device)

    if shift:
        emb_cls, emb_sub = get_embeddings_shift(
                image, model, pretrained_path=pretrained_path,
                device=device)
    else:
        emb_cls, emb_sub = get_embeddings(
                image, model, pretrained_path=pretrained_path,
                device=device)
    embs = dict(cls=emb_cls, sub=emb_sub)

    img_size = image.shape[0]
    emb_size = embs['cls'][0].shape[0]
    tile_size = img_size // emb_size
    assert img_size % emb_size == 0

    # TODO: check consistency with shifting
    embs['rgb'] = np.stack([
            reduce(
                image[..., i].astype(np.float16) / 255.0,
                '(h1 h) (w1 w) -> h1 w1', 'mean',
                h=tile_size, w=tile_size).astype(np.float32)
            for i in range(3)])
    return embs


class HistologyExtractor:

    tile_sizes = {'high': 256, 'low': 16}

    def __init__(self, pretrained_path=None):
        self.pretrained_path = pretrained_path

    def extract(
            self, image, shift=True, use_pretrained=True, device='cuda'):
        if use_pretrained and (self.pretrained_path is not None):
            pretrained_path = self.pretrained_path
        else:
            pretrained_path = None
        embs = extract_raw(
                image, shift=shift,
                pretrained_path=pretrained_path, device=device)
        embs = smoothen_embeddings(embs, device=device)

        return embs
