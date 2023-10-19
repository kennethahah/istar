import sys

from .run import run
from .utility.io import load_toml


def cli():
    pass


config_file = sys.argv[1]

config = load_toml(config_file)

slide_paths = {
    name: slide['data']
    for name, slide in config['slides'].items()}

# slide_options = {
#     name: slide['options'] if 'options' in slide else {}
#     for name, slide in config['slides'].items()
# }

analyses = {
    name: (settings['type'], settings['options'])
    for name, settings in config['analyses'].items()
}

run(
    slide_paths=slide_paths,
    model_config=config['istar'],
    optim_config=config['optimization'],
    analyses=analyses,
    )
