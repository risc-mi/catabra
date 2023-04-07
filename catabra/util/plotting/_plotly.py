#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Union

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio

from catabra.util.plotting import _common


def _get_color(i: int, template: Optional[str] = None) -> str:
    colorway = pio.templates[template or pio.templates.default]['layout']['colorway']
    return ','.join([str(c) for c in px.colors.hex_to_rgb(colorway[i % len(colorway)])])


def training_history(x: np.ndarray, ys, title: Optional[str] = 'Training History', legend=None, text=None):
    """
    Plot training history, with timestamps on x- and metrics on y-axis.
    :param x: Timestamps, array of shape `(n,)`.
    :param ys: Metrics, array of shape `(n,)`.
    :param title: The title of the figure.
    :param legend: Names of the individual curves. None or a list of strings with the same length as `ys`.
    :param text: Additional information to display when hovering over a point. Same for all metrics. None or a list of
    strings with the same length as `x`.
    :return: plotly figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    fig = go.Figure()

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
        if isinstance(text, str):
            text = [text]
        assert len(legend) == len(ys)
        assert text is None or len(text) == len(x)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        for y, lbl in zip(ys, legend):
            fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode='lines+markers', text=text))
        maxes = [y.abs().max() for y in ys]
        maxes = [np.log10(m) for m in maxes if m > 0]
        if min(maxes) + 1 < max(maxes):
            # logarithmic scale
            # Unfortunately, plotly does not support "symlog", and workarounds are hacky.
            #   (https://community.plotly.com/t/unable-to-see-negative-log-axis-scatter-plot/1364/3)
            y_scale = 'log'

    fig.update_layout(
        xaxis=dict(title='Time' + unit_suffix),
        yaxis=dict(title='Metric', type=y_scale),
        title=dict(text=_common.make_title(None, title), x=0.5, xref='paper'),
        showlegend=len(ys) > 1 or any(lbl is not None for lbl in legend)
    )
    return fig


def regression_scatter(y_true: np.ndarray, y_hat: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                       name: Optional[str] = None, title: Optional[str] = 'Truth-Prediction Plot', cmap='Blues'):
    """
    Create a scatter plot with results of a regression task.
    :param y_true: Ground truth, array of shape `(n,)`. May contain NaN values.
    :param y_hat: Predictions, array of shape `(n,)`. Must have same data type as `y_true`, and may also contain NaN
    values.
    :param sample_weight: Sample weights.
    :param name: Name of the target variable.
    :param title: The title of the resulting figure.
    :param cmap: The color map for coloring points according to `sample_weight`.
    :return: plotly figure object.
    """
    assert len(y_true) == len(y_hat)
    assert y_true.dtype == y_hat.dtype

    if y_true.dtype.kind == 'm':
        unit, uom = _common.convert_timedelta(y_true)
        y_true = y_true / unit
        y_hat = y_hat / unit
        unit_suffix = f' [{uom}]'
    else:
        unit_suffix = ''

    d_min = min(y_true.min(), y_hat.min())
    d_max = max(y_true.max(), y_hat.max())
    if isinstance(y_true, pd.Series):
        text = y_true.index
        y_true = y_true.values
    else:
        text = None
    if isinstance(y_hat, pd.Series):
        y_hat = y_hat.values
    if sample_weight is None:
        df = pd.DataFrame(data=dict(y_true=y_true, y_hat=y_hat))
        if text is not None:
            df['ID'] = text
            text = 'ID'
        fig = px.scatter(df, x='y_true', y='y_hat', text=text, marginal_x='histogram', marginal_y='histogram')
        fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
    else:
        def _to_color(_c: tuple) -> str:
            return f'rgb({round(_c[0] * 256)},{round(_c[1] * 256)},{round(_c[2] * 256)})'

        cmap = _common.get_colormap(cmap)
        if text is None:
            text = ['y_true={}<br>y_hat={}<br>sample_weight={}'.format(y_true[j], y_hat[j], sample_weight[j])
                    for j in range(len(y_true))]
        else:
            text = ['ID={}<br>y_true={}<br>y_hat={}<br>sample_weight={}'.format(text[j], y_true[j], y_hat[j],
                                                                                sample_weight[j])
                    for j in range(len(y_true))]
        colors = cmap((sample_weight - sample_weight.min()) / (sample_weight.max() - sample_weight.min()))
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=y_true,
                y=y_hat,
                mode='markers',
                marker=dict(color=[_to_color(c) for c in colors]),
                hovertemplate='%{text}',
                text=text,
                name=''
            )
        )
    fig.add_shape(type='line', line=dict(dash='dash', width=0.5), x0=d_min, y0=d_min, x1=d_max, y1=d_max)

    fig.update_layout(
        title=dict(text=_common.make_title(name, title, sep='<br>'), x=0.5, xref='paper'),
        xaxis=dict(title='True label' + unit_suffix),
        yaxis=dict(scaleanchor='x', title='Predicted label' + unit_suffix)
    )
    return fig


def confusion_matrix(cm: pd.DataFrame, name: Optional[str] = None, title: Optional[str] = 'Confusion Matrix',
                     cmap='Blues'):
    """
    Plot a confusion matrix.
    :param cm: The confusion matrix to plot. A DataFrame whose rows correspond to true classes and whose columns
    correspond to predicted classes. If the last class is called "__total__", it is assumed to contain row- or
    column totals.
    :param name: Name of the target variable.
    :param title: The title of the resulting figure.
    :param cmap: The color map.
    :return: plotly figure object.
    """

    # This implementation is not very clean. It was copied from the MC3 data pipeline and modified.

    if len(cm) > 2 and cm.index[-1] == '__total__':
        cm = cm.iloc[:-1]
    if len(cm.columns) > 2 and cm.columns[-1] == '__total__':
        cm = cm.iloc[:, :-1]
    n = cm.shape[0]
    classes = cm.columns
    matrix = cm.values

    n_samples = cm.sum().sum()
    div = cm.sum(axis=1).values
    cm_rel = cm.values / np.maximum(1, div[..., np.newaxis])
    thresh = (cm_rel.max() + cm_rel.min()) / 2.0

    fig = go.Figure()
    coords = [i + 0.5 for i in range(n + 1)]
    mg = np.meshgrid(coords[:-1], coords[1:])
    x = list(mg[0].flatten())
    y = list(mg[1].flatten())
    z = matrix[::-1].flatten()

    pred_totals = np.sum(matrix, axis=0)
    true_totals = np.sum(matrix, axis=1)
    part_acc = [100 * matrix[i, i] / max(1, s) for i, s in enumerate(pred_totals)] + \
               [100 * matrix[i, i] / max(1, s) for i, s in enumerate(true_totals)] + \
               [100 * np.trace(matrix) / n_samples]
    part_totals = list(pred_totals) + list(true_totals) + [n_samples]

    x += coords[:-1] + [coords[-1]] * (n + 1)
    y += coords[:1] * n + coords[1:][::-1] + coords[:1]

    mg = np.meshgrid(classes, classes)
    x_class = mg[0].flatten()
    y_class = mg[1][::-1].flatten()

    text = ['<b>{}</b>'.format((np.round(t, 2), 100 * t / n_samples) for t in z)]
    text += ['<b>{}</b><br>{:.2f}%'.format(np.round(t, 2), a) if t > 0 else '<b>{}</b>'.format(np.round(t, 2))
             for t, a in zip(part_totals, part_acc)]

    if n == 2:
        correct_names = ['NPV', 'PPV', 'Specificity', 'Sensitivity']
    else:
        correct_names = ['Correct'] * n * 2
    correct_names.append('Accuracy')
    hovertext = [f'True label: {j}<br>Predicted label: {i}<br>Count: {k}'
                 for i, j, k in zip(x_class, y_class, z)]
    hovertext += [('<b>{}</b><br>Count: {}<br>{}: {:.2f}%' if t > 0 else '<b>{}</b><br>Count: {}').format(
        'Total' if i == 2 * n else
        f'Predicted label: {classes[i]}' if i < n else f'True label: {classes[i - n]}', t, cn, a)
        for i, (t, a, cn) in enumerate(zip(part_totals, part_acc, correct_names))]

    def _to_color(_c: tuple) -> str:
        return f'rgb({round(_c[0] * 256)},{round(_c[1] * 256)},{round(_c[2] * 256)})'

    cmap = _common.get_colormap(cmap)
    cmap_min = _to_color(cmap(0.0))
    cmap_max = _to_color(cmap(1.0))

    text_color = [cmap_max if t < thresh * max(1, div[n - i // n - 1]) else cmap_min for i, t in enumerate(z)]

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        text=text,
        hovertext=hovertext,
        hoverinfo='text',
        mode='text',
        textfont=dict(color=text_color + ['White'] * (2 * n + 1))
    ))

    # Set axes properties
    fig.update_xaxes(range=[0, n + 1], showgrid=False)
    fig.update_yaxes(range=[0, n + 1], showgrid=False)

    shapes = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            shapes.append(
                go.layout.Shape(
                    type='rect',
                    x0=j,
                    y0=n + 1 - i,
                    x1=j + 1,
                    y1=n - i,
                    line=dict(
                        width=1,
                        color='White'
                    ),
                    fillcolor=_to_color(cmap(val) if div is None else cmap(val / max(1, div[i]))),
                    layer='below'
                )
            )
    for i in range(len(classes)):
        shapes.append(
            go.layout.Shape(
                type='rect',
                x0=i,
                y0=0,
                x1=i + 1,
                y1=1,
                line=dict(width=1, color='White'),
                fillcolor='DarkGray',
                layer='below'
            )
        )
        shapes.append(
            go.layout.Shape(
                type='rect',
                x0=n,
                y0=i + 1,
                x1=n + 1,
                y1=i + 2,
                line=dict(width=1, color='White'),
                fillcolor='DarkGray',
                layer='below'
            )
        )
    shapes.append(
        go.layout.Shape(
            type='rect',
            x0=n,
            y0=0,
            x1=n + 1,
            y1=1,
            line=dict(width=1, color='White'),
            fillcolor='Gray',
            layer='below'
        )
    )

    fig.update_layout(
        xaxis=dict(
            title='Predicted label',
            ticktext=classes,
            tickvals=coords[:-1],
            showline=False,
            zeroline=False,
            constrain='domain',
        ),
        yaxis=dict(
            title='True label',
            ticktext=list(reversed(classes)),
            tickvals=coords[1:],
            scaleanchor='x',
            scaleratio=1,
            showline=False,
            zeroline=False
        ),
        shapes=shapes,
        hovermode='closest',
        title=dict(text=_common.make_title(name, title, sep='<br>'), x=0.5, xref='paper')
    )

    return fig


def roc_pr_curve(xs, ys, deviation=None, name: Optional[str] = None, title: Optional[str] = 'auto', roc: bool = True,
                 legend=None, deviation_legend=None, positive_prevalence: float = -1.):
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
    :param roc: Whether ROC- or PR curves are plotted.
    :param legend: Names of the individual curves. None or a list of string with the same length as `xs` and `ys`.
    :param deviation_legend: Names of the deviation curves. None or a list of strings with the same length as `ys`.
    :param positive_prevalence: Prevalence of the positive class. Only relevant for PR curves.
    :return: plotly figure object.
    """
    if not isinstance(xs, list):
        xs = [xs]
    if not isinstance(ys, list):
        ys = [ys]
    assert len(xs) == len(ys)
    fig = go.Figure()
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
        if deviation_legend is None:
            deviation_legend = [None] * len(ys)
        elif isinstance(deviation_legend, str):
            deviation_legend = [deviation_legend]
        assert len(deviation) == len(ys)
        assert len(deviation_legend) == len(ys)
        assert len(legend) == len(ys)
        assert all(x.dtype.kind == xs[0].dtype.kind for x in xs)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        x_min = min(x.min() for x in xs)
        x_max = max(x.max() for x in xs)
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        if roc:
            fig.add_shape(type='line', line=dict(dash='dash', width=0.5), x0=0, y0=0, x1=1, y1=1)
        elif positive_prevalence >= 0.:
            # unfortunately, we cannot rely on `ys[0][0]` being the positive prevalence
            fig.add_shape(type='line', line=dict(dash='dash', width=0.5), x0=0, y0=positive_prevalence, x1=1,
                          y1=positive_prevalence)
        for i, (x, y, dv, lbl, dv_lbl) in enumerate(zip(xs, ys, deviation, legend, deviation_legend)):
            color = _get_color(i)
            if dv is not None:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([dv[1], dv[0, ::-1]]),
                    fill='toself',
                    fillcolor='rgba(' + color + ',0.2)',
                    line_color='rgba(255,255,255,0)',
                    name=dv_lbl
                ))
            fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode='lines', line=dict(color='rgb(' + color + ')')))
    else:
        x_min = x_max = y_min = y_max = 0

    if title == 'auto':
        title = 'Receiver Operating Characteristic' if roc else 'Precision-Recall Curve'
    fig.update_layout(
        xaxis=dict(title='False positive rate' if roc else 'Recall', range=[min(-0.05, x_min), max(1.05, x_max)],
                   constrain='domain'),
        yaxis=dict(title='True positive rate' if roc else 'Precision', scaleanchor='x',
                   range=[min(-0.05, y_min), max(1.05, y_max)], constrain='domain'),
        title=dict(text=_common.make_title(name, title, sep='<br>'), x=0.5, xref='paper'),
        showlegend=len(xs) > 1 or any(lbl is not None for lbl in legend) or
                   any(lbl is not None for lbl in deviation_legend)
    )
    return fig


def threshold_metric_curve(th: np.ndarray, ys, threshold: Optional[float] = None, name: Optional[str] = None,
                           title: Optional[str] = 'Threshold-Metric Plot', legend=None):
    """
    Plot threshold-vs.-metric curves, with thresholds on the x- and corresponding thresholded metrics on the y-axis.
    :param th: The thresholds, a single array of shape `(n,)`.
    :param ys: The y-coordinates of the curve(s), either a single array or a list of arrays of shape `(n,)`.
    :param threshold: Actual decision threshold used for making predictions, or None. If given, a dashed vertical line
    is drawn to indicate the threshold.
    :param name: Name of the target variable.
    :param title: The title of the figure.
    :param legend: Names of the individual curves. None or a list of string with the same length as `ys`.
    :return: plotly figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    fig = go.Figure()
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
        if threshold is not None:
            fig.add_shape(type='line', line=dict(dash='dash', width=0.5),
                          x0=threshold, y0=min(0, y_min), x1=threshold, y1=max(1, y_max))
        for y, lbl in zip(ys, legend):
            fig.add_trace(go.Scatter(x=th, y=y, name=lbl, mode='lines'))
    else:
        x_min = x_max = y_min = y_max = 0

    fig.update_layout(
        xaxis=dict(title='Threshold', range=[min(-0.05, x_min), max(1.05, x_max)], constrain='domain'),
        yaxis=dict(title='Metric', range=[min(-0.05, y_min), max(1.05, y_max)], constrain='domain'),
        title=dict(text=_common.make_title(name, title, sep='<br>'), x=0.5, xref='paper'),
        showlegend=len(ys) > 1 or any(lbl is not None for lbl in legend)
    )
    return fig


def calibration_curve(th_lower: np.ndarray, th_upper: np.ndarray, ys, name: Optional[str] = None, deviation=None,
                      title: Optional[str] = 'Calibration Curve', legend=None, deviation_legend=None):
    """
    Plot calibration curves.
    :param th_lower: Lower/left ends of threshold bins, array of shape `(n,)`.
    :param th_upper: Upper/right ends of threshold bins, array of shape `(n,)`.
    :param ys: y-coordinates of the curve(s), either a single array or a list of arrays of shape `(n,)`.
    :param deviation: y-coordinates of deviations of the curve(s), indicating errors, standard deviations, confidence
    intervals, and the like. None or a list of arrays of shape `(2, n)`.
    :param name: Name of the target variable.
    :param title: The title of the figure.
    :param legend: Names of the individual curves. None or a list of strings with the same length as `ys`.
    :param deviation_legend: Names of the deviation curves. None or a list of strings with the same length as `ys`.
    :return: plotly figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    fig = go.Figure()
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
        if deviation_legend is None:
            deviation_legend = [None] * len(ys)
        elif isinstance(deviation_legend, str):
            deviation_legend = [deviation_legend]
        assert len(deviation) == len(ys)
        assert len(deviation_legend) == len(ys)
        assert len(legend) == len(ys)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        x = (th_lower + th_upper) * 0.5
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        for i, (y, dv, lbl, dv_lbl) in enumerate(zip(ys, deviation, legend, deviation_legend)):
            color = _get_color(i)
            if dv is not None:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([dv[1], dv[0, ::-1]]),
                    fill='toself',
                    fillcolor='rgba(' + color + ',0.2)',
                    line_color='rgba(255,255,255,0)',
                    name=dv_lbl
                ))
            fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode='lines+markers', line=dict(color='rgb(' + color + ')')))
    else:
        y_min = y_max = 0

    fig.update_layout(
        xaxis=dict(title='Threshold', constrain='domain'),
        yaxis=dict(title='Fraction of positive class', range=[min(-0.05, y_min), max(1.05, y_max)], constrain='domain'),
        title=dict(text=_common.make_title(name, title, sep='<br>'), x=0.5, xref='paper'),
        showlegend=len(ys) > 1 or any(lbl is not None for lbl in legend)
                   or any(lbl is not None for lbl in deviation_legend)
    )
    return fig


def beeswarm(values: pd.DataFrame, colors: Union[pd.DataFrame, pd.Series, None] = None,
             color_name: Optional[str] = None, title: Optional[str] = None, x_label: Optional[str] = None,
             cmap='red_blue', **kwargs):
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
    :param cmap: Name of a color map.
    :return: plotly figure object.
    """

    assert all(values[c].dtype.kind in 'ifb' for c in values.columns)
    n_samples, n_features = values.shape
    row_height = 0.4

    colors_orig = colors
    if isinstance(colors, pd.DataFrame):
        assert len(colors) == len(values)
    elif isinstance(colors, pd.Series):
        assert len(colors) == len(values)
        if colors.dtype.name == 'category':
            colors = colors.cat.codes

    cmap = _common.get_colormap(cmap)
    fig = go.Figure()

    values_min = values.min().min()
    values_max = values.max().max()

    # beeswarm
    for pos, column in enumerate(reversed(values.columns)):
        fig.add_shape(type='line', line=dict(dash='dash', color='#cccccc', width=0.5),
                      x0=values_min, y0=pos, x1=values_max, y1=pos)
        if isinstance(colors, pd.Series):
            col = colors
            col_orig = colors_orig
        elif isinstance(colors, pd.DataFrame) and column in colors.columns:
            col = colors[column]
            col_orig = colors_orig[column]
        else:
            col = None
            col_orig = None
        gray, colored = _common.beeswarm_feature(values[column], col, row_height)

        if gray is not None:
            x_gray, y_gray, i_gray = gray
            if colors is None:
                c_gray = (cmap(np.zeros((1,), dtype=np.float32))[0, :3] * 255).astype(np.uint8)
                c_gray = [f'rgb({c_gray[0]},{c_gray[1]},{c_gray[2]})'] * len(x_gray)
            else:
                c_gray = ['#777777'] * len(x_gray)
        if colored is not None:
            x_colored, y_colored = colored[:2]
            c_colored = (cmap((np.clip(colored[2], colored[3], colored[4]) - colored[3]) /
                              (colored[4] - colored[3]))[:, :3] * 255).astype(np.uint8)
            c_colored = [f'rgb({c[0]},{c[1]},{c[2]})' for c in c_colored]

        if gray is None:
            if colored is not None:
                x = x_colored
                y = y_colored
                c = c_colored
                i = colored[5]
        elif colored is None:
            x = x_gray
            y = y_gray
            c = c_gray
            i = i_gray
        else:
            x = pd.concat([x_gray, x_colored])
            y = np.concatenate([y_gray, y_colored])
            c = c_gray + c_colored
            i = np.concatenate([i_gray, colored[5]])

        if col_orig is None:
            text = ['ID={}<br>X={}'.format(x.index[j], x.iloc[j]) for j in range(len(x))]
        else:
            text = ['ID={}<br>X={}<br>{}={}'.format(x.index[j], x.iloc[j], color_name, col_orig.iloc[i[j]])
                    for j in range(len(x))]

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=pos + y,
                mode='markers',
                marker=dict(size=4, color=c),
                hovertemplate='%{text}',
                text=text,
                name=column
            )
        )

    fig.update_layout(
        xaxis=dict(title=x_label, constrain='domain'),
        yaxis=dict(
            ticktext=list(reversed(values.columns)),
            tickvals=list(range(n_features)),
            zeroline=False
        ),
        title=dict(text=_common.make_title(None, title), x=0.5, xref='paper'),
        showlegend=False
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(range=[-1, n_features], showgrid=False)

    return fig


def horizontal_bar(values: Union[pd.Series, pd.DataFrame], groups: Optional[dict] = None, title: Optional[str] = None,
                   x_label: Optional[str] = None, **kwargs):
    """
    Create a horizontal bar plot.
    :param values: Values to plot, a DataFrame whose rows correspond to the rows in the plot and whose columns
    correspond to grouped bars.
    :param groups: Optional, grouping of columns. If given, group names must be mapped to lists of columns in `values`.
    For instance, if `values` has columns "label1_>0", "label1_<0" and "label2", then `groups` might be set to

        {
            "label1": ["label1_>0", "label1_<0"],
            "label2": ["label2"]
        }

    :param title: Title of the figure.
    :param x_label: Label of the x-axis.
    :return: plotly figure object.
    """

    n_features = len(values)
    if isinstance(values, pd.Series):
        values = values.to_frame()
    if groups is None:
        groups = {c: [c] for c in values.columns}
    else:
        assert len(groups) >= 1
        assert all(all(c in values.columns for c in g) for g in groups.values())

    fig = go.Figure()

    neg_color = (_common.get_colormap('blue_rgb')[:3] * 255).astype(np.uint8)
    pos_color = (_common.get_colormap('red_rgb')[:3] * 255).astype(np.uint8)
    pos_color = f'rgb({pos_color[0]},{pos_color[1]},{pos_color[2]})'
    neg_color = f'rgb({neg_color[0]},{neg_color[1]},{neg_color[2]})'

    # draw the bars
    for i, (g, columns) in enumerate(groups.items()):
        for c in columns:
            fig.add_trace(
                go.Bar(
                    name=g,
                    orientation='h',
                    x=values[c].iloc[::-1],
                    y=values.index[::-1],
                    offsetgroup=i,
                    marker=dict(
                        color=[neg_color if values[c].iloc[j] < 0 else pos_color for j in range(n_features - 1, -1, -1)]
                    )
                )
            )

    fig.update_layout(
        xaxis=dict(title=x_label, constrain='domain'),
        title=dict(text=_common.make_title(None, title), x=0.5, xref='paper'),
        showlegend=False
    )

    return fig
