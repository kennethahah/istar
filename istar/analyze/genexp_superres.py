from .analyze import Analysis, _register_analysis


def _genexp_superres(slidebag, output_path, **kwargs):
    slidebag.visualize_genexp_superres(output_path)


_register_analysis(
    name='genexp_superres',
    analysis=Analysis(
        description=(
            'Perferm super-resolution '
            'of gene expression measurements'),
        function=_genexp_superres,
    ),
)
