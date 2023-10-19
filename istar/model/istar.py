from .extract import HistologyExtractor
from .predict import GenExpPredictor
from .generate import GenExpGenerator3d


class Istar:

    def __init__(
            self, extractor_kwargs={}, predictor_kwargs={},
            generator_kwargs={}):
        self.extractor = HistologyExtractor(**extractor_kwargs)
        self.predictor = GenExpPredictor(**predictor_kwargs)
        self.generator = GenExpGenerator3d(**generator_kwargs)
