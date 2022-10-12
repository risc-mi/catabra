from pathlib import Path
from typing import Dict, Union, Optional, List

import pandas as pd

from catabra.base import logging
from catabra.base.config import Invocation
from catabra.util import plotting


class EvaluationConfig:

    @property
    def bootstrapping_repetitions(self) -> int:
        return self._bootstrapping_repetitions

    @property
    def copy_data(self) -> Union[bool, int, float]:
        return self._copy_data

    @property
    def static_plots(self) -> bool:
        return self._static_plots

    @property
    def interactive_plots(self) -> bool:
        return self._interactive_plots

    def __init__(self, src: Dict):
        self._bootstrapping_repetitions = src.get('bootstrapping_repetitions', 0)
        self._copy_data = src.get('copy_evaluation_data', False)

        self._static_plots = src.get('static_plots', True)
        self._interactive_plots = src.get('interactive_plots', False)
        if self._interactive_plots and plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            self._interactive_plots = False

        # TODO
        # self._metrics = src.get(encoder.task_ + '_metrics', [])


class EvaluationInvocation(Invocation):

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def model_id(self):
        return self._model_id

    @property
    def explain(self):
        return self._explain

    @property
    def glob(self) -> Optional[bool]:
        return self._glob

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @property
    def bootstrapping_repetitions(self) -> Optional[int]:
        return self._bootstrapping_repetitions

    @property
    def bootstrapping_metrics(self) -> Optional[List]:
        return self._bootstrapping_metrics

    def __init__(
        self,
        *table,
        split: Optional[str] = None,
        sample_weight: Optional[str] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None,
        folder: Union[str, Path] = None,
        model_id=None,
        explain=None,
        glob: Optional[bool] = False,
        batch_size: Optional[int] = None,
        threshold: Optional[float] = None,
        bootstrapping_repetitions: Optional[int] = None,
        bootstrapping_metrics: Optional[list] = None,
    ):

        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)
        self._folder = folder
        self._model_id = model_id
        self._explain = explain
        self._glob = glob
        self._batch_size = batch_size
        self._threshold = threshold
        self._bootstrapping_repetitions = bootstrapping_repetitions
        self._bootstrapping_metrics = bootstrapping_metrics

    def update(self, src: Dict):
        super().update(src)
        if self._folder is None:
            self._folder = src.get('folder')
        if self._model_id is None:
            self._model_id = src.get('model_id')
        if self._explain is None:
            self._explain = src.get('explain')
        if self._glob is None:
            self._glob = src.get('glob')
        if self._threshold is None:
            self._threshold = src.get('threshold')
        if self._bootstrapping_repetitions is None:
            self._bootstrapping_repetitions = src.get('bootstrapping_repetitions')
        if self._bootstrapping_metrics is None:
            self._bootstrapping_metrics = src.get('bootstrapping_metrics')
        if self._batch_size is None:
            self._batch_size = src.get('bootstrapping_metrics')

        if self._threshold is None:
            self._threshold = 0.5

    def to_dict(self) -> Dict:
        dic = super().to_dict()
        dic.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            folder=self._folder,
            model_id=self._model_id,
            explain=self._explain,
            glob=self._glob,
            split=self._split,
            sample_weight=self._sample_weight,
            out=self._out,
            jobs=self._jobs,
            threshold=self._threshold,
            bootstrapping_repetitions=self._bootstrapping_repetitions,
            bootstrapping_metrics=self._bootstrapping_metrics,
            timestamp=self._start
        ))
        return dic

