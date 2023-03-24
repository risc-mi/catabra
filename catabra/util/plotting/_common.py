#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def make_title(name: Optional[str], title: Optional[str], sep: str = '\n') -> Optional[str]:
    if title is None:
        return name
    elif name is None:
        return title
    else:
        return title + sep + name


def get_colormap(cmap):
    if isinstance(cmap, str):
        try:
            from matplotlib import pyplot as plt
            return plt.get_cmap(cmap)
        except:     # noqa
            pass
        try:
            import shap
            return getattr(shap.plots.colors, cmap)
        except:     # noqa
            pass
        raise ValueError('Unknown color map: ' + cmap)
    else:
        return cmap


def convert_timedelta(x: np.ndarray) -> Tuple[pd.Timedelta, str]:
    mean = np.abs(x).mean()
    unit = pd.Timedelta(365.2525, unit='d')
    if mean > unit:
        uom = 'y'
    else:
        unit = pd.Timedelta(1, unit='d')
        if mean > unit:
            uom = 'd'
        else:
            unit = pd.Timedelta(1, unit='h')
            if mean > unit:
                uom = 'h'
            else:
                unit = pd.Timedelta(1, unit='m')
                if mean > unit:
                    uom = 'm'
                else:
                    unit = pd.Timedelta(1, unit='s')
                    uom = 's'
    return unit, uom


def beeswarm_feature(values: pd.Series, colors: Optional[pd.Series], row_height: float):
    n_bins = 100
    n_samples = len(values)
    inds = np.random.permutation(n_samples)
    values = values.iloc[inds]
    if colors is None:
        colored_feature = False
    else:
        colors = colors.iloc[inds]
        if colors.dtype.name == 'category':
            colored_feature = False
        else:
            colored_feature = True
            try:
                colors = np.array(colors, dtype=np.float64)  # make sure this can be numeric
            except:  # noqa
                colored_feature = False

    quant = (n_bins * (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)).round().values
    layer = 0
    last_bin = -1
    ys = np.zeros(n_samples)
    for ind in np.argsort(quant + np.random.randn(n_samples) * 1e-6):
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    ys *= 0.9 * (row_height / np.max(ys + 1))

    if colored_feature:
        # trim the color range, but prevent the color range from collapsing
        vmin = np.nanpercentile(colors, 5)
        vmax = np.nanpercentile(colors, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(colors, 1)
            vmax = np.nanpercentile(colors, 99)
            if vmin == vmax:
                vmin = np.min(colors)
                vmax = np.max(colors)
        if vmin > vmax:  # fixes rare numerical precision issues
            vmin = vmax

        nan_mask = np.isnan(colors)
        cvals = colors[np.invert(nan_mask)].copy()
        cvals[cvals > vmax] = vmax
        cvals[cvals < vmin] = vmin

        return ((values[nan_mask], ys[nan_mask], inds[nan_mask]) if nan_mask.any() else None), \
            (values[np.invert(nan_mask)], ys[np.invert(nan_mask)], cvals, vmin, vmax, inds[np.invert(nan_mask)])
    else:
        return (values, ys, inds), None
