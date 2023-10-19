from .analyze import Analysis, _register_analysis
from ..utility.visual import plot_matrix


def plot_comparison(slidebag, model):
    slide = list(slidebag.slides.values())[0]
    y = slide.st.cnts_latent_superres
    _, locs = slidebag.flatten_spatial()
    locs = locs.reshape(y.shape[:2] + (3,))
    y_pred = model.generator.predict(locs)
    c = 0
    plot_matrix(y[..., c], 'y.png')
    plot_matrix(y_pred[..., c], 'y_pred.png')


def _generate_genexp_3d(slidebag, model, cache, output_path, **kwargs):
    model.generator.train(
            slidebag=slidebag, cache=cache,
            load_saved=True, prefix=output_path,
            **kwargs)

    plot_comparison(slidebag, model)

    slidebag.generate_genexp_3d(
            model.generator, output_path+'data.pickle')


_register_analysis(
    name='generate_genexp_3d',
    analysis=Analysis(
        description='Generate 3D gene expression volume',
        function=_generate_genexp_3d,
    ),
)
