from typing import Optional
import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go

from . import _common


def regression_scatter(y_true: np.ndarray, y_hat: np.ndarray, name: Optional[str] = None,
                       title: Optional[str] = 'Truth-Prediction Plot'):
    """
    Create a scatter plot with results of a regression task.
    :param y_true: Ground truth, array of shape `(n,)`. May contain NaN values.
    :param y_hat: Predictions, array of shape `(n,)`. Must have same data type as `y_true`, and may also contain NaN
    values.
    :param name: Name of the target variable.
    :param title: The title of the resulting figure.
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
    df = pd.DataFrame(data=dict(y_true=y_true, y_hat=y_hat))
    if text is not None:
        df['ID'] = text
        text = 'ID'
    fig = px.scatter(df, x='y_true', y='y_hat', text=text, marginal_x='histogram', marginal_y='histogram')
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
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
    cm_rel = cm.values / div[..., np.newaxis]
    thresh = (cm_rel.max() + cm_rel.min()) / 2.0

    fig = go.Figure()
    coords = [i + 0.5 for i in range(n + 1)]
    mg = np.meshgrid(coords[:-1], coords[1:])
    x = list(mg[0].flatten())
    y = list(mg[1].flatten())
    z = matrix[::-1].flatten()

    pred_totals = np.sum(matrix, axis=0)
    true_totals = np.sum(matrix, axis=1)
    part_acc = [100 * matrix[i, i] / s for i, s in enumerate(pred_totals)] + \
               [100 * matrix[i, i] / s for i, s in enumerate(true_totals)] + \
               [100 * np.trace(matrix) / n_samples]
    part_totals = list(pred_totals) + list(true_totals) + [n_samples]

    x += coords[:-1] + [coords[-1]] * (n + 1)
    y += coords[:1] * n + coords[1:][::-1] + coords[:1]

    mg = np.meshgrid(classes, classes)
    x_class = mg[0].flatten()
    y_class = mg[1][::-1].flatten()

    text = ['<b>{}</b>'.format(np.round(t, 2), 100 * t / n_samples) for t in z]
    text += ['<b>{}</b><br>{:.2f}%'.format(np.round(t, 2), a) for t, a in zip(part_totals, part_acc)]

    hovertext = [f'True label: {j}<br>Predicted label: {i}<br>Count: {k}'
                 for i, j, k in zip(x_class, y_class, z)]
    hovertext += ['<b>{}</b><br>Count: {}<br>Correct: {:.2f}%'.format(
        'Total' if i == 2 * n else
        f'Predicted label: {classes[i]}' if i < n else f'True label: {classes[i - n]}', t, a)
        for i, (t, a) in enumerate(zip(part_totals, part_acc))]

    def _to_color(_c: tuple) -> str:
        return f'rgb({round(_c[0] * 256)},{round(_c[1] * 256)},{round(_c[2] * 256)})'

    if isinstance(cmap, str):
        from matplotlib import pyplot as plt
        cmap = plt.get_cmap(cmap)
    cmap_min = _to_color(cmap(0.0))
    cmap_max = _to_color(cmap(1.0))

    text_color = [cmap_max if t < thresh * div[n - i // n - 1] else cmap_min for i, t in enumerate(z)]

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
                    fillcolor=_to_color(cmap(val) if div is None else cmap(val / div[i])),
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


def roc_pr_curve(xs, ys, name: Optional[str] = None, title: Optional[str] = 'auto', roc: bool = True,
                 legend=None, positive_prevalence: float = -1.):
    """
    Plot ROC or PR curve(s).
    :param xs: The x-coordinates of the curve(s), either a single array or a list of arrays. In the case of ROC curves
    they correspond to the false positive rates, in the case of PR curves they correspond to recall.
    :param ys: The y-coordinates of the curve(s), either a single array or a list of arrays. In the case of ROC curves
    they correspond to the true positive rates, in the case of PR curves they correspond to precision.
    :param name: Name of the target variable.
    :param title: The title of the figure. If "auto", the title is either "Receiver Operating Characteristic" or
    "Precision-Recall Curve" depending on the value of `roc`.
    :param roc: Whether ROC- or PR curves are plotted.
    :param legend: Names of the individual curves. None or a list of string with the same length as `xs` and `ys`.
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
        for x, y, lbl in zip(xs, ys, legend):
            fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode='lines'))
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
        showlegend=len(xs) > 1 or any(lbl is not None for lbl in legend)
    )
    return fig


def threshold_metric_curve(th: np.ndarray, ys, name: Optional[str] = None,
                           title: Optional[str] = 'Threshold-Metric Plot', legend=None):
    """
    Plot threshold-vs.-metric curves, with thresholds on the x- and corresponding thresholded metrics on the y-axis.
    :param th: The thresholds, a single array of shape `(n,)`.
    :param ys: The y-coordinates of the curve(s), either a single array or a list of arrays of shape `(n,)`.
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
        if (0 <= th).all() and (th <= 1).all():
            fig.add_shape(type='line', line=dict(dash='dash', width=0.5),
                          x0=0.5, y0=min(0, y_min), x1=0.5, y1=max(1, y_max))
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


def calibration_curve(th_lower: np.ndarray, th_upper: np.ndarray, ys, name: Optional[str] = None,
                      title: Optional[str] = 'Calibration Curve', legend=None):
    """
    Plot calibration curves.
    :param th_lower: Lower/left ends of threshold bins, array of shape `(n,)`.
    :param th_upper: Upper/right ends of threshold bins, array of shape `(n,)`.
    :param ys: y-coordinates of the curve(s), either a single array or a list of arrays of shape `(n,)`.
    :param name: Name of the target variable.
    :param title: The title of the figure.
    :param legend: Names of the individual curves. None or a list of string with the same length as `ys`.
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
        assert len(legend) == len(ys)
        assert all(y.dtype.kind == ys[0].dtype.kind for y in ys)
        x = (th_lower + th_upper) * 0.5
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        for y, lbl in zip(ys, legend):
            fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode='lines+markers'))
    else:
        y_min = y_max = 0

    fig.update_layout(
        xaxis=dict(title='Threshold', constrain='domain'),
        yaxis=dict(title='Fraction of positive class', range=[min(-0.05, y_min), max(1.05, y_max)], constrain='domain'),
        title=dict(text=_common.make_title(name, title, sep='<br>'), x=0.5, xref='paper'),
        showlegend=len(ys) > 1 or any(lbl is not None for lbl in legend)
    )
    return fig
