import os
import pickle

from PIL import Image
import numpy as np
import tomli
import pandas as pd


Image.MAX_IMAGE_PIXELS = None


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename: str, verbose: bool = True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_toml(filename: str, verbose: bool = True):
    with open(filename, 'r') as f:
        content = tomli.loads(f.read())
    return content


def load_dataframe(
        filename: str, index: bool = True,
        header: bool = True, verbose: bool = True):

    index_col = None
    if index:
        index_col = 0

    header_row = None
    if header:
        header_row = 0

    df = pd.read_csv(
            filename, sep='\t',
            header=header_row, index_col=index_col)

    if verbose:
        print(f'Dataframe loaded from {filename}')

    return df


def save_dataframe(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines
