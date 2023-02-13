from typing import Optional
from pathlib import Path

from ...util import io
from ...util.plotting import plotly_backend
from ..base import TrainingMonitorBackend


class PlotlyBackend(TrainingMonitorBackend):

    @classmethod
    def name(cls) -> str:
        return 'plotly'

    def __init__(self, update_interval: int = 5, **kwargs):
        super(PlotlyBackend, self).__init__(**kwargs)
        self._update_interval = int(update_interval)
        if self._update_interval <= 0:
            self._meta = ''
        else:
            self._meta = f'http-equiv="refresh" content="{self._update_interval}" '

    @property
    def update_interval(self) -> int:
        return self._update_interval

    def get_info(self) -> Optional[str]:
        return 'Open the following file for monitoring the training progress: ' + \
            (Path(self._folder) / 'live.html').as_posix()

    def launch(self):
        self._save_html(Path(self._folder) / 'live.html', [], self._meta)

    def shutdown(self):
        src = Path(self._folder)

        if (src / 'live.json').exists():
            obj = io.load(src / 'live.json')
            try:
                (src / 'live.json').unlink()
            except:  # noqa
                pass
            self._save_html(src / 'live.html', obj, '')     # automatic updates are not needed any more
        else:
            try:
                (src / 'live.html').unlink()
            except:  # noqa
                pass

    def set_params(self, **params):
        pass

    def _update(self, event, timestamp: float, elapsed_time: float, step: int, text: Optional[str], **metrics: float):
        src = Path(self._folder)

        if (src / 'live.json').exists():
            obj = io.load(src / 'live.json')
        else:
            obj = []

        obj.append(
            dict(event=event, timestamp=timestamp, elapsed_time=elapsed_time, step=step, text=text, metrics=metrics)
        )
        io.dump(obj, src / 'live.json')

        self._save_html(src / 'live.html', obj, self._meta)

    @staticmethod
    def _save_html(file, obj: list, meta: str):
        if not obj:
            with open(file, mode='wt', encoding='utf-8') as _f:
                _f.write(
                    f'<html><head><meta charset="utf-8" {meta}/></head><body>'
                    'Nothing to display yet.'
                    '</body></html>'
                )
        elif plotly_backend is not None:
            all_metrics = {m for o in obj for m in o['metrics']}
            if all_metrics:
                fig = plotly_backend.go.Figure()
                for m in all_metrics:
                    sub = [o for o in obj if m in o['metrics']]
                    x = [s['elapsed_time'] for s in sub]
                    y = [s['metrics'][m] for s in sub]
                    t = [s['text'] for s in sub]
                    fig.add_trace(plotly_backend.go.Scatter(x=x, y=y, name=m, mode='lines+markers', text=t))

                fig.update_layout(
                    xaxis=dict(title='Time [s]', fixedrange=len(meta) > 0),
                    yaxis=dict(title='Metric', fixedrange=len(meta) > 0),
                    title=dict(text='Training History', x=0.5, xref='paper'),
                    showlegend=True,
                    hovermode='x'
                )

                header = f'<html>\n<head><meta charset="utf-8" {meta}/></head>\n<body>\n'
                footer = '</body>\n</html>'
                html = header + fig.to_html(include_plotlyjs='cdn', include_mathjax='cdn', full_html=False) + footer
                with open(file, mode='wt', encoding='utf-8') as _f:
                    _f.write(html)
