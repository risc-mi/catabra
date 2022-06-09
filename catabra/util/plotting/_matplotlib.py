from typing import Optional, Union
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import _common


def training_history(x: np.ndarray, ys, title: Optional[str] = 'Training History', ax=None, figsize='auto',
                     legend=None, **kwargs):
    """
    Plot training history, with timestamps on x- and metrics on y-axis.
    :param x: Timestamps, array of shape `(n,)`.
    :param ys: Metrics, array of shape `(n,)`.
    :param title: The title of the figure.
    :param ax: An existing axis, or None.
    :param figsize: Figure size.
    :param legend: Names of the individual curves. None or a list of string with the same length as `ys`.
    :return: Matplotlib figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    if ax is None:
        if figsize == 'auto':
            figsize = (np.log10(len(x) + 1) + 5, 5)
        _, ax = plt.subplots(figsize=figsize)

    if x.dtype.kind == 'm':
        unit, uom = _common.convert_timedelta(x)
        x = x / unit
        unit_suffix = f' [{uom}]'
    else:
        unit_suffix = ''

    y_scale = 'linear'
    if ys:
        assert all(len(x) == len(y) for y in ys)
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        assert len(legend) == len(ys)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        for y, lbl in zip(ys, legend):
            ax.plot(x, y, label=lbl, marker='.')
        if any(lbl is not None for lbl in legend):
            ax.legend(loc='best')
        maxes = [y.abs().max() for y in ys]
        maxes = [np.log10(m) for m in maxes if m > 0]
        if min(maxes) + 1 < max(maxes):
            # logarithmic scale
            y_scale = 'symlog' if any((y < 0).any() for y in ys) else 'log'

    ax.set(xlabel='Time' + unit_suffix, ylabel='Metric', yscale=y_scale, title=_common.make_title(None, title))
    plt.close(ax.figure)
    return ax.figure


def regression_scatter(y_true: np.ndarray, y_hat: np.ndarray, name: Optional[str] = None,
                       title: Optional[str] = 'Truth-Prediction Plot', ax=None, figsize='auto'):
    """
    Create a scatter plot with results of a regression task, using the default Matplotlib backend.
    :param y_true: Ground truth, array of shape `(n,)`. May contain NaN values.
    :param y_hat: Predictions, array of shape `(n,)`. Must have same data type as `y_true`, and may also contain NaN
    values.
    :param name: Name of the target variable.
    :param title: The title of the resulting figure.
    :param ax: An existing axis, or None.
    :param figsize: Figure size. If "auto", the 'optimal' figure size is determined automatically.
    :return: Matplotlib figure object.
    """
    assert len(y_true) == len(y_hat)
    assert y_true.dtype == y_hat.dtype
    if ax is None:
        if figsize == 'auto':
            figsize = int(round(np.log10(len(y_true) + 1))) + 3
            figsize = (figsize, figsize)
        _, ax = plt.subplots(figsize=figsize)

    if y_true.dtype.kind == 'm':
        unit, uom = _common.convert_timedelta(y_true)
        y_true = y_true / unit
        y_hat = y_hat / unit
        unit_suffix = f' [{uom}]'
    else:
        unit_suffix = ''

    d_min = min(y_true.min(), y_hat.min())
    d_max = max(y_true.max(), y_hat.max())
    ax.set_aspect(1.)
    ax.plot([d_min, d_max], [d_min, d_max], ls='--', c='gray', lw=1.)
    ax.scatter(y_true, y_hat, marker='.', rasterized=len(y_true) > 500)

    ax.set(
        ylabel='Predicted label' + unit_suffix,
        xlabel='True label' + unit_suffix,
        title=_common.make_title(name, title)
    )
    plt.close(ax.figure)
    return ax.figure


def confusion_matrix(cm: pd.DataFrame, name: Optional[str] = None, title: Optional[str] = 'Confusion Matrix',
                     cmap='Blues', ax=None, figsize='auto'):
    """
    Plot a confusion matrix with Matplotlib.
    :param cm: The confusion matrix to plot. A DataFrame whose rows correspond to true classes and whose columns
    correspond to predicted classes. If the last class is called "__total__", it is assumed to contain row- or
    column totals.
    :param name: Name of the target variable.
    :param title: The title of the resulting figure.
    :param cmap: The color map.
    :param ax: An existing axis, or None.
    :param figsize: Figure size. If "auto", the 'optimal' figure size is determined automatically.
    :return: Matplotlib figure object.
    """
    if cm.index[-1] != '__total__':
        cm = pd.concat([cm, pd.DataFrame(data={c: cm[c].sum() for c in cm.columns}, index=['__total__'])],
                       axis=0, sort=False)
    if cm.columns[-1] != '__total__':
        cm['__total__'] = cm.sum(axis=1)
    assert len(cm) == len(cm.columns)
    n = len(cm) - 1

    if ax is None:
        if figsize == 'auto':
            figsize = (n + 2) if n < 7 else n + 1
            figsize = (figsize, figsize)
        _, ax = plt.subplots(figsize=figsize)

    cm_rel = 2 * cm.values / cm.values.sum(axis=1, keepdims=True)
    cm_rel[:, -1] = 0.95
    cm_rel[-1] = 0.95
    cm_rel[-1, -1] = 1.
    im = ax.imshow(cm_rel, interpolation='nearest', cmap=cmap)
    thresh = (cm_rel.max() + cm_rel.min()) / 2.0
    fmt = '.2f' if cm.iloc[:, 0].dtype.kind == 'f' else 'd'

    cmap_min, cmap_max = im.cmap(0.0), im.cmap(1.0)

    for i in range(n):
        for j in range(n):
            text_cm = format(cm.iloc[i, j], fmt)        # + '\n' + format(100 * cm_rel[i, j], '.2f') + '%'
            color = cmap_max if cm_rel[i, j] < thresh else cmap_min
            ax.text(j, i, text_cm, ha='center', va='center', color=color, weight='bold')

        text_cm = format(cm.iloc[i, -1], fmt) + '\n' + format(100 * cm_rel[i, i], '.2f') + '%'
        ax.text(n, i, text_cm, ha='center', va='center', color=cmap_min, weight='bold')

    for j in range(n):
        text_cm = format(cm.iloc[-1, j], fmt) + '\n' + format(100 * cm.iloc[j, j] / cm.iloc[-1, j], '.2f') + '%'
        ax.text(j, n, text_cm, ha='center', va='center', color=cmap_min, weight='bold')

    text_cm = format(cm.iloc[-1, -1], fmt) + '\n' + \
        format(100 * cm.values[:-1, :-1].trace() / cm.iloc[-1, -1], '.2f') + '%'
    ax.text(n, n, text_cm, ha='center', va='center', color=cmap_min, weight='bold')

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=cm.columns[:-1],
        yticklabels=cm.index[:-1],
        ylabel='True label',
        xlabel='Predicted label',
        title=_common.make_title(name, title)
    )
    ax.set_ylim((n + 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='45' if any(len(str(c)) > 10 for c in cm.columns) else 'horizontal')
    plt.close(ax.figure)
    return ax.figure


def roc_pr_curve(xs, ys, deviation=None, name: Optional[str] = None, title: Optional[str] = 'auto', ax=None, figsize=(5, 5),
                 roc: bool = True, legend=None, positive_prevalence: float = -1., **kwargs):
    """
    Plot ROC or PR curve(s).
    :param xs: The x-coordinates of the curve(s), either a single array or a list of arrays. In the case of ROC curves
    they correspond to the false positive rates, in the case of PR curves they correspond to recall.
    :param ys: The y-coordinates of the curve(s), either a single array or a list of arrays. In the case of ROC curves
    they correspond to the true positive rates, in the case of PR curves they correspond to precision.
    :param deviation: y-coordinates of deviations of the curve(s), indicating errors, standard deviations, confidence
    intervals, and the like. None or a list of arrays of shape `(2, n)`.
    :param name: Name of the target variable.
    :param title: The title of the figure. If "auto", the title is either "Receiver Operating Characteristic" or
    "Precision-Recall Curve" depending on the value of `roc`.
    :param ax: An existing axis, or None.
    :param figsize: Figure size.
    :param roc: Whether ROC- or PR curves are plotted.
    :param legend: Names of the individual curves. None or a list of string with the same length as `xs` and `ys`.
    :param positive_prevalence: Prevalence of the positive class. Only relevant for PR curves.
    :return: Matplotlib figure object.
    """
    if not isinstance(xs, list):
        xs = [xs]
    if not isinstance(ys, list):
        ys = [ys]
    assert len(xs) == len(ys)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if xs:
        assert all(len(x) == len(y) for x, y in zip(xs, ys))
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        if deviation is None:
            deviation = [None] * len(ys)
        elif not isinstance(deviation, list):
            deviation = [deviation]
        assert len(deviation) == len(ys)
        assert len(legend) == len(ys)
        assert all(x.dtype.kind == xs[0].dtype.kind for x in xs)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        x_min = min(x.min() for x in xs)
        x_max = max(x.max() for x in xs)
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        ax.set_xlim(min(-0.05, x_min), max(1.05, x_max))
        ax.set_ylim(min(-0.05, y_min), max(1.05, y_max))
        ax.set_aspect(1.)
        if roc:
            ax.plot([0, 1], [0, 1], ls='--', c='gray', lw=1.)
        elif positive_prevalence >= 0.:
            # unfortunately, we cannot rely on `ys[0][0]` being the positive prevalence
            ax.plot([0, 1], [positive_prevalence, positive_prevalence], ls='--', c='gray', lw=1.)
        for x, y, dv, lbl in zip(xs, ys, deviation, legend):
            line = ax.plot(x, y, label=lbl)[0]
            if dv is not None:
                ax.fill_between(x, dv[0], dv[1], alpha=0.2, facecolor=line.get_color())
        if any(lbl is not None for lbl in legend):
            ax.legend(loc='lower right' if roc else 'best')

    if title == 'auto':
        title = 'Receiver Operating Characteristic' if roc else 'Precision-Recall Curve'
    ax.set(
        xlabel='False positive rate' if roc else 'Recall',
        ylabel='True positive rate' if roc else 'Precision',
        title=_common.make_title(name, title)
    )
    plt.close(ax.figure)
    return ax.figure


def threshold_metric_curve(th: np.ndarray, ys, name: Optional[str] = None,
                           title: Optional[str] = 'Threshold-Metric Plot', ax=None, figsize=(10, 5), legend=None):
    """
    Plot threshold-vs.-metric curves, with thresholds on the x- and corresponding thresholded metrics on the y-axis.
    :param th: The thresholds, a single array of shape `(n,)`.
    :param ys: The y-coordinates of the curve(s), either a single array or a list of arrays of shape `(n,)`.
    :param name: Name of the target variable.
    :param title: The title of the figure.
    :param ax: An existing axis, or None.
    :param figsize: Figure size.
    :param legend: Names of the individual curves. None or a list of string with the same length as `ys`.
    :return: Matplotlib figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if ys:
        assert all(len(th) == len(y) for y in ys)
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        assert len(legend) == len(ys)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        x_min = th.min()
        x_max = th.max()
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        ax.set_xlim(min(-0.05, x_min), max(1.05, x_max))
        ax.set_ylim(min(-0.05, y_min), max(1.05, y_max))
        if (0 <= th).all() and (th <= 1).all():
            ax.axvline(0.5, ls='--', c='gray', lw=1.)
        for y, lbl in zip(ys, legend):
            ax.plot(th, y, label=lbl)
        if any(lbl is not None for lbl in legend):
            ax.legend(loc='best')

    ax.set(
        xlabel='Threshold',
        ylabel='Metric',
        title=_common.make_title(name, title)
    )
    plt.close(ax.figure)
    return ax.figure


def calibration_curve(th_lower: np.ndarray, th_upper: np.ndarray, ys, name: Optional[str] = None, deviation=None,
                      title: Optional[str] = 'Calibration Curve', ax=None, figsize=(5, 5), legend=None, **kwargs):
    """
    Plot calibration curves.
    :param th_lower: Lower/left ends of threshold bins, array of shape `(n,)`.
    :param th_upper: Upper/right ends of threshold bins, array of shape `(n,)`.
    :param ys: y-coordinates of the curve(s), either a single array or a list of arrays of shape `(n,)`.
    :param deviation: y-coordinates of deviations of the curve(s), indicating errors, standard deviations, confidence
    intervals, and the like. None or a list of arrays of shape `(2, n)`.
    :param name: Name of the target variable.
    :param title: The title of the figure.
    :param ax: An existing axis, or None.
    :param figsize: Figure size.
    :param legend: Names of the individual curves. None or a list of strings with the same length as `ys`.
    :return: Matplotlib figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if ys:
        assert len(th_lower) == len(th_upper)
        assert all(len(th_lower) == len(y) for y in ys)
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        if deviation is None:
            deviation = [None] * len(ys)
        elif not isinstance(deviation, list):
            deviation = [deviation]
        assert len(deviation) == len(ys)
        assert len(legend) == len(ys)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        x = (th_lower + th_upper) * 0.5
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        ax.set_ylim(min(-0.05, y_min), max(1.05, y_max))
        for y, dv, lbl in zip(ys, deviation, legend):
            line = ax.plot(x, y, label=lbl, marker='.')[0]
            if dv is not None:
                ax.fill_between(x, dv[0], dv[1], alpha=0.2, facecolor=line.get_color())
        if any(lbl is not None for lbl in legend):
            ax.legend(loc='best')

    ax.set(
        xlabel='Threshold',
        ylabel='Fraction of positive class',
        title=_common.make_title(name, title)
    )
    plt.close(ax.figure)
    return ax.figure


def beeswarm(values: pd.DataFrame, colors: Union[pd.DataFrame, pd.Series, None] = None,
             color_name: Optional[str] = None, title: Optional[str] = None, x_label: Optional[str] = None, ax=None,
             figsize='auto', cmap='red_blue', **kwargs):
    """
    Create a beeswarm plot, inspired by (and largely copied from) the shap package [1]. This plot type is very useful
    for displaying local feature importance scores for a set of samples.

    [1] https://github.com/slundberg/shap/blob/master/shap/plots/_beeswarm.py

    :param values: Values to plot, a DataFrame whose columns correspond to the rows in the plot.
    :param colors: Optional, colors of the points. A DataFrame with the same shape and column names as `values`, or a
    Series with the same number of rows as `values`.
    :param color_name: Name of the color bar; only relevant if `colors` is provided.
    :param title: Title of the figure.
    :param x_label: Label of the x-axis.
    :param ax: An existing axis, or None.
    :param figsize: Figure size or "auto".
    :param cmap: Name of a color map.
    :return: Matplotlib figure object.
    """

    assert all(values[c].dtype.kind in 'ifb' for c in values.columns)
    n_samples, n_features = values.shape
    row_height = 0.4
    if ax is None:
        if figsize == 'auto':
            figsize = n_features * row_height + 1.5
            figsize = (8, figsize)
        _, ax = plt.subplots(figsize=figsize)

    color_ticks = ['Low', 'High']
    if isinstance(colors, pd.DataFrame):
        assert colors.shape == values.shape
        assert (colors.columns == values.columns).all()
    elif isinstance(colors, pd.Series):
        assert len(colors) == len(values)
        if colors.dtype.name == 'category':
            color_ticks = colors.cat.categories
            colors = colors.cat.codes

    cmap = _common.get_colormap(cmap)
    ax.axvline(x=0, color='#999999', zorder=-1)

    # beeswarm
    for pos, column in enumerate(reversed(values.columns)):
        ax.axhline(y=pos, color='#cccccc', lw=0.5, dashes=(1, 5), zorder=-1)
        if isinstance(colors, pd.DataFrame):
            col = colors[column]
        elif isinstance(colors, pd.Series):
            col = colors
        else:
            col = None
        gray, colored = _common.beeswarm_feature(values[column], col, row_height)

        if gray is not None:
            if col is None:
                col = cmap(0.)
            else:
                col = '#777777'
            ax.scatter(gray[0], pos + gray[1], s=16, alpha=1, linewidth=0, zorder=3, color=col,
                       rasterized=n_samples > 500)
        if colored is not None:
            ax.scatter(colored[0], pos + colored[1], cmap=cmap, c=colored[2], vmin=colored[3], vmax=colored[4],
                       s=16, linewidth=0, alpha=1, zorder=3, rasterized=n_samples > 500)

    # color bar
    if colors is not None:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap)
        m.set_array([0, 1])
        cb = plt.colorbar(m, ticks=np.linspace(0, 1, len(color_ticks)), aspect=1000)
        cb.set_ticklabels(color_ticks)
        if color_name is not None:
            cb.set_label(color_name, size=12, labelpad=0)
        cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params('y', length=20, width=0.5, which='major')

    ax.set(
        xlabel=x_label,
        yticks=range(n_features),
        yticklabels=reversed(values.columns),
        ylim=(-1, n_features),
        title=_common.make_title(None, title)
    )
    plt.close(ax.figure)
    return ax.figure
