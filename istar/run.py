import os
from typing import Dict, Tuple, Any, Optional
import warnings

from .data.slide import get_data
from .model import Istar
from .analyze import analyses as _analyses


def run(
        slide_paths: Dict[str, str],
        model_config: Optional[Dict[str, Any]] = None,
        optim_config: Optional[Dict[str, Any]] = None,
        analyses: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
        ):
    r"""Train iStar model and run analyses"""

    if analyses is None:
        analyses = {}

    cache = model_config['cache']
    output_path = model_config['output_path']

    extractor_kwargs = {
            'pretrained_path': model_config['extractor_path']}
    predictor_kwargs = {
            'n_states': model_config['n_states'],
            'model_path': output_path + 'models/predictor.pickle'}
    generator_kwargs = {
            'n_states': 1,  # TODO: pass from analysis config
            'model_path': output_path + 'models/generator.pickle'}
    model = Istar(
            extractor_kwargs=extractor_kwargs,
            predictor_kwargs=predictor_kwargs,
            generator_kwargs=generator_kwargs)

    slidebag = get_data(
            slide_paths=slide_paths, config=model_config,
            extractor=model.extractor, cache=cache)
    model.predictor.train(
            slidebag=slidebag, cache=cache, **optim_config)

    # TODO: move this to analyses
    slidebag.enhance_genexp_resolution(
            model=model.predictor, cache=cache)

    analysis_path = os.path.join(output_path, 'analyses')
    for name, (analysis_type, options) in analyses.items():
        if analysis_type in _analyses:
            output_path = os.path.join(analysis_path, name) + '/'
            _analyses[analysis_type].function(
                    model=model, slidebag=slidebag,
                    cache=cache, output_path=output_path,
                    **options)
        else:
            warnings.warn(f'Unknown analysis `{analysis_type}`')
