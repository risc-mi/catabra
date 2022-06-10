from typing import Union

from . import _matplotlib as mpl_backend
from ..io import Path, make_path

try:
    from . import _plotly as plotly_backend
except ModuleNotFoundError:
    plotly_backend = None


PLOTLY_WARNING = 'plotly is required for creating interactive plots. You can install it either through' \
                 ' pip (`pip install plotly==5.7.0`) or conda (`conda install plotly=5.7.0 -c plotly`).' \
                 ' Visit https://plotly.com/python/ for details.'


def save(fig, fn: Union[str, Path], png: bool = False):
    """
    Save a figure or a list of figures to disk.
    :param fig: The figure(s) to save. May be a Matplotlib figure object, a plotly figure object, or a dict whose
    values are such figure objects.
    :param fn: The file or directory. It is recommended to leave the file extension unspecified and simply pass
    "/path/to/figure" instead of "/path/to/figure.png". The file extension is then determined automatically depending
    on the type of `fig` and on the value of `png`. If `fig` is a dict, `fn` refers to the parent directory.
    :param png: Whether to save Matplotlib figures as PNG or as PDF. Ignored if a file extension is specified in `fn`
    or if `fig` is a plotly figure, which are always saved as HTML.
    """
    fn = make_path(fn)
    if isinstance(fig, dict):
        fn.mkdir(parents=True, exist_ok=True)
        if all(hasattr(f, 'to_html') for f in fig.values()):
            header = '<html>\n<head><meta charset="utf-8" /></head>\n<body>\n<div>'
            footer = '</body>\n</html>'
            for name, f in fig.items():
                links = '  |  '.join([_n if _n == name else f'<a href="{_n}.html">{_n}</a>' for _n in fig])
                html = header + links + '</div>\n' + \
                    f.to_html(include_plotlyjs='cdn', include_mathjax='cdn', full_html=False) + footer
                with open(fn / (name + '.html'), mode='wt', encoding='utf-8') as _f:
                    _f.write(html)
        else:
            for name, f in fig.items():
                save(f, fn / name, png=png)
    elif hasattr(fig, 'savefig'):
        if fn.suffix == '':
            fn = fn.with_suffix('.png' if png else '.pdf')
        fig.savefig(fn, bbox_inches='tight')
    elif hasattr(fig, 'write_html'):
        if fn.suffix == '':
            fn = fn.with_suffix('.html')
        fig.write_html(fn, include_plotlyjs='cdn', include_mathjax='cdn', full_html=True)
    else:
        raise ValueError(f'Cannot save figure of type {type(fig)}.')