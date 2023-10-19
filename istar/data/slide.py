import os

import numpy as np
import pandas as pd

from ..utility.io import (
        load_image, save_image,
        load_toml, load_dataframe, save_dataframe,
        load_pickle, save_pickle, read_lines)
from ..utility.image import rescale_image, pad_image
from ..utility.image import resize as resize_image
from ..utility.visual import plot_matrix, plot_spots, make_gif, mat_to_img
from ..data.spotpatch import get_spot_mask, get_patches_flat
from ..data.threedim import get_pixels_locs_3d


class SlideST:

    def __init__(self, cnts, locs, radius):
        self.cnts = cnts
        self.locs = locs
        self.radius = radius

    def rescale(self, scale):
        self.locs = (self.locs * scale).round().astype(int)
        self.radius = self.radius * scale

    def load_locs(self, path):
        self.locs = load_dataframe(path)

    def save_locs(self, path):
        save_dataframe(self.locs, path)

    def keep_genes(self, names):
        self.cnts = self.cnts[names]


class SlideHistology:

    def __init__(self, image):
        self.image = image
        self.embs = None

    def rescale(self, scale, skip_image=False):
        dtype = self.image.dtype
        image = self.image.astype('float32')
        self.image = rescale_image(image, scale).astype(dtype)

    def pad_image(self, size, value):
        self.image = pad_image(self.image, size=size, value=value)

    def extract(self, model, shift):
        self.embs = model.extract(self.image, shift=shift)

    def tensorize_embs(self):
        embs = self.embs
        embs = np.concatenate([
            embs['cls'], embs['sub'], embs['rgb']])
        embs = embs.transpose(1, 2, 0)
        self.embs = embs

    def load_embeddings(self, path):
        self.embs = load_pickle(path)

    def save_embeddings(self, path):
        save_pickle(self.embs, path)

    def load_image(self, path):
        self.image = load_image(path)

    def save_image(self, path):
        save_image(self.image, path)


class Slide:

    def __init__(self, path: str, cache: bool = True):

        paths = {}
        paths['cache'] = {
                'locs': 'cache/spot-locations.tsv',
                'histology_image': 'cache/histology.jpg',
                'histology_embeddings': 'cache/histology-embeddings.pickle',
                'cnts_superres': 'cache/genexp-counts-superres.pickle',
                'cnts_latent_superres': 'cache/genexp-embeddings.pickle',
                }

        paths['inputs'] = {
                'params': 'params.toml',
                'histology_image': 'histology.jpg',
                'cnts': 'genexp-counts.tsv',
                'locs': 'spot-locations.tsv',
                }

        paths = {
                name: {
                    nam: os.path.join(path, suff)
                    for nam, suff in ps.items()}
                for name, ps in paths.items()}
        self.paths = paths

        histology_img = load_image(paths['inputs']['histology_image'])
        self.histology = SlideHistology(histology_img)

        params = load_toml(paths['inputs']['params'])
        pixel_size = params['pixel_size']
        if 'z' in params.keys():
            z_loc = params['z'] / pixel_size
        self.pixel_size = pixel_size
        self.z_loc = z_loc

        cnts = load_dataframe(paths['inputs']['cnts'])
        locs = load_dataframe(paths['inputs']['locs'])
        self.st = SlideST(
                cnts=cnts, locs=locs, radius=params['spot_radius'])

    def rescale(self, target_pixel_size, skip_image=False):
        scale = self.pixel_size / target_pixel_size
        if not skip_image:
            self.histology.rescale(scale)
        self.st.rescale(scale)
        self.z_loc = self.z_loc * scale
        self.pixel_size = target_pixel_size

    def preprocess(
            self, target_pixel_size, padding_size, padding_value,
            cache):

        path = self.paths['cache']['histology_image']
        exist = os.path.exists(path)
        load_image = cache and exist
        save_image = cache and (not exist)
        self.rescale(target_pixel_size, skip_image=load_image)
        if load_image:
            self.histology.load_image(path)
        else:
            self.histology.pad_image(
                    size=padding_size, value=padding_value)
        if save_image:
            self.histology.save_image(path)

    def extract_histology(self, model, shift, cache):
        path = self.paths['cache']['histology_embeddings']
        exist = os.path.exists(path)
        if cache and exist:
            self.histology.load_embeddings(path)
        else:
            self.histology.extract(model, shift=shift)
            self.histology.tensorize_embs()

        if cache and not exist:
            self.histology.save_embeddings(path)
        self.histology.embs2img_factor = model.tile_sizes['low']

    def enhance_genexp_resolution(self, model, cache):

        path_genexp = self.paths['cache']['cnts_superres']
        path_latent = self.paths['cache']['cnts_latent_superres']
        exists = os.path.exists(path_genexp) and os.path.exists(path_latent)
        if cache and exists:
            self.load_genexp_superres(path_genexp)
            self.load_genexp_latent_superres(path_latent)
        else:
            genexp, latent = model.predict(self.histology.embs)
            self.st.cnts_superres = genexp
            self.st.cnts_latent_superres = latent

        if cache and not exists:
            self.save_genexp_superres(path_genexp)
            self.save_genexp_latent_superres(path_latent)

    def save_genexp_superres(self, path):
        save_pickle(self.st.cnts_superres, path)

    def save_genexp_latent_superres(self, path):
        save_pickle(self.st.cnts_latent_superres, path)

    def load_genexp_superres(self, path):
        self.st.cnts_superres = load_pickle(path)

    def load_genexp_latent_superres(self, path):
        self.st.cnts_latent_superres = load_pickle(path)

    def visualize_genexp_superres(self, path):
        genes = self.st.cnts.columns.to_list()
        for i, g in enumerate(genes):
            x = self.st.cnts_superres[..., i]
            p = os.path.join(path, g) + '.png'
            plot_matrix(x, p)

    def visualize_genexp_spotlevel(self, genes, path):
        if genes == 'all':
            genes = self.st.cnts.columns.to_list()
        for g in genes:
            cnts = self.st.cnts[g]
            locs = self.st.locs[['y', 'x']].astype(int).to_numpy()
            img = self.histology.image
            radius = int(self.st.radius)
            outfile = os.path.join(path, g) + '.png'
            plot_spots(
                    img=img, cnts=cnts, locs=locs, radius=radius,
                    cmap='turbo', weight=0.5,
                    outfile=outfile)

    def patchify(
            self, circular_spot=True, normalize=False):
        locs = self.st.locs
        cnts = self.st.cnts
        factor = self.histology.embs2img_factor

        embs = self.histology.embs

        locs = locs[['y', 'x']].to_numpy()

        locs = (locs / factor).round().astype('int')
        radius = self.st.radius / factor

        mask = get_spot_mask(radius, circular=circular_spot)
        embs_patches = get_patches_flat(embs, locs, mask)
        out = {
                'embs': embs_patches, 'cnts': cnts,
                'mask_size': mask.sum()}
        return out

    def keep_genes(self, names):
        self.st.keep_genes(names)


class SlideBag:

    def __init__(self, slides):
        self.slides = slides

    def preprocess(self, *args, **kwargs):
        for slide in self.slides.values():
            slide.preprocess(*args, **kwargs)

    def extract_histology(self, *args, **kwargs):
        for slide in self.slides.values():
            slide.extract_histology(*args, **kwargs)

    def enhance_genexp_resolution(self, *args, **kwargs):
        for slide in self.slides.values():
            slide.enhance_genexp_resolution(*args, **kwargs)

    def save_genexp_superres(self, *args, **kwargs):
        for slide in self.slides.values():
            slide.save_genexp_superres(*args, **kwargs)

    def save_genexp_latent_superres(self, *args, **kwargs):
        for slide in self.slides.values():
            slide.save_genexp_latent_superres(*args, **kwargs)

    def visualize_genexp_superres(self, path):
        for name, slide in self.slides.items():
            slide.visualize_genexp_superres(os.path.join(path, name))

    def visualize_genexp_spotlevel(self, genes, path):
        for name, slide in self.slides.items():
            slide.visualize_genexp_spotlevel(
                    genes, os.path.join(path, name))

    def patchify(self, normalize=True, *args, **kwargs):
        out_list = [
                slide.patchify(*args, **kwargs)
                for slide in self.slides.values()]
        embs_list = [out['embs'] for out in out_list]
        cnts_list = [out['cnts'] for out in out_list]
        mask_size = out_list[0]['mask_size']
        embs = np.concatenate(embs_list)
        cnts = pd.concat(
                cnts_list, join='inner', ignore_index=True)

        merged = {
                'embs': embs, 'cnts': cnts, 'mask_size': mask_size}
        return merged

    def select_genes(self, n_top=None, priority_path=None):
        cnts_list = [slide.st.cnts for slide in self.slides.values()]
        cnts = pd.concat(
                cnts_list, join='inner', ignore_index=True)
        order = cnts.var().to_numpy().argsort()[::-1]
        names = cnts.columns.to_list()
        names_all = [names[i] for i in order]

        names_top = names_all

        if n_top is not None:
            names_top = names_top[:n_top]

        if priority_path is None:
            names_priority = []
        else:
            names_priority = read_lines(priority_path)
            names_priority = [
                    name for name in names_priority
                    if (name in names_all) and (name not in names_top)]

        names = names_priority + names_top

        for slide in self.slides.values():
            slide.keep_genes(names)

    def flatten_spatial(self, discrete=False):
        slides = list(self.slides.values())
        values_slides = [
                s.st.cnts_latent_superres for s in slides]
        zlocs_slides = [
                s.z_loc for s in slides]
        values, locs = get_pixels_locs_3d(
                values_slides, zlocs_slides)
        if not discrete:
            locs = locs.astype('float32')
            locs -= locs.min(0)
            locs /= locs.max() + 1e-12
        return values, locs

    def generate_genexp_3d(self, model, output_path):
        values_train, locs_train = self.flatten_spatial(discrete=True)
        locs = fill_grid(locs_train)
        locs = locs.astype('float32')
        locs -= locs.min(tuple(range(locs.ndim-1)))
        locs /= locs.max() + 1e-12
        plot_3d(model, locs, output_path)
        breakpoint()
        # self.cnts_latent_3d = latent
        # TODO: save latent

    def interpolate(self, gene, path):
        slides = list(self.slides.values())
        z = [s.z_loc for s in slides]
        z_size = max(z) - min(z) + 1
        is_latent = gene == 'genexp-embeddings'
        if is_latent:
            x = [s.st.cnts_latent_superres for s in slides]
            x = np.stack(x, 2)
            target_shape = x.shape[:2] + (z_size,) + x.shape[3:]
        else:
            i = slides[0].st.cnts.columns.to_list().index(gene)
            x = [s.st.cnts_superres[..., i] for s in slides]
            x = np.stack(x, 2)
            target_shape = x.shape[:2] + (z_size,)
        x = resize_image(x, target_shape)
        save_pickle(x, path+'.pickle')
        if not is_latent:
            for axis in range(3):
                img = mat_to_img(x)
                make_gif(img, f'{path}-axis{axis}.gif', axis=axis)
        breakpoint()

    def save_genexp_3d(self, path):
        save_pickle(self.st.cnts_3d, path)

    def load_genexp_3d(self, path):
        self.st.cnts_3d = load_pickle(path)


def get_data(slide_paths, config, extractor, cache):
    slides = {
        slide_name: Slide(slide_paths[slide_name], cache=cache)
        for slide_name in slide_paths.keys()
    }

    slidebag = SlideBag(slides)

    slidebag.preprocess(
            target_pixel_size=config['target_pixel_size'],
            padding_size=extractor.tile_sizes['high'],
            padding_value=config['padding_value'],
            cache=cache)
    slidebag.extract_histology(
            extractor, shift=config['shift'],
            cache=cache)
    if 'priority_genes_path' in config.keys():
        priority_genes_path = config['priority_genes_path']
    else:
        priority_genes_path = None
    slidebag.select_genes(
            n_top=config['n_top_genes'],
            priority_path=priority_genes_path)
    return slidebag


def fill_grid(locs):
    locs_min = locs.min(0)
    locs_max = locs.max(0)
    ranges = [
            range(lmin, lmax+1)
            for lmin, lmax in zip(locs_min, locs_max)]
    grid = np.meshgrid(*ranges, indexing='ij')
    grid = np.stack(grid, -1)
    return grid


def plot_3d(model, locs, prefix):
    c = 0  # channel to plot
    for axis in range(3):
        size = locs.shape[axis]
        i = size // 2  # midpoint in axis
        x = locs.take(indices=i, axis=axis)
        y = model.predict(x)
        plot_matrix(y[..., c], prefix + f'-{axis}.png')
