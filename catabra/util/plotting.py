#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

from typing import Union

from catabra_lib.plotting import mpl_backend, plotly_backend

from catabra.util.io import Path, make_path

PLOTLY_WARNING = 'plotly is required for creating interactive plots. You can install it either through' \
                 ' pip (`pip install plotly==5.7.0`) or conda (`conda install plotly=5.7.0 -c plotly`).' \
                 ' Visit https://plotly.com/python/ for details.'


def save(fig, fn: Union[str, Path], png: bool = False):
    """
    Save a figure or a list of figures to disk.

    Parameters
    ----------
    fig:
        The figure(s) to save. May be a Matplotlib figure object, a plotly figure object, or a dict whose values are
        such figure objects.
    fn: str | Path
        The file or directory. It is recommended to leave the file extension unspecified and simply pass
        `"/path/to/figure"` instead of `"/path/to/figure.png"`. The file extension is then determined automatically
        depending on the type of `fig` and on the value of `png`. If `fig` is a dict, `fn` refers to the parent
        directory.
    png: bool, default=False
        Whether to save Matplotlib figures as PNG or as PDF. Ignored if a file extension is specified in `fn` or if
        `fig` is a plotly figure, which are always saved as HTML.
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
        if fn.suffix not in ('.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg',
                             '.svgz', '.tif', '.tiff'):
            fn = fn.parent / (fn.name + ('.png' if png else '.pdf'))
        fig.savefig(fn, bbox_inches='tight')
    elif hasattr(fig, 'write_html'):
        if fn.suffix != '.html':
            fn = fn.parent / (fn.name + '.html')
        fig.write_html(fn, include_plotlyjs='cdn', include_mathjax='cdn', full_html=True)
    else:
        raise ValueError(f'Cannot save figure of type {type(fig)}.')


__all__ = ['mpl_backend', 'plotly_backend', 'save']
