from .analyze import Analysis, _register_analysis


def _visualize_genexp_spotlevel(slidebag, output_path, genes, **kwargs):
    slidebag.visualize_genexp_spotlevel(genes, output_path)


_register_analysis(
    name='visualize_genexp_spotlevel',
    analysis=Analysis(
        description=(
            'Visualize spot-level gene expression measurements'),
        function=_visualize_genexp_spotlevel,
    ),
)
