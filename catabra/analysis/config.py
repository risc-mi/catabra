from pathlib import Path
from typing import Dict, Union, Optional, List, Iterable
import pandas as pd

from ..util import io
from ..core.base import Invocation


class AnalysisInvocation(Invocation):

    @property
    def classify(self) -> Optional[Iterable[Union[str, Path, pd.DataFrame]]]:
        return self._classify

    @property
    def regress(self) -> Optional[Iterable[Union[str, Path, pd.DataFrame]]]:
        return self._regress

    @property
    def target(self) -> Optional[Iterable[str]]:
        return self._target

    @property
    def group(self) -> Optional[str]:
        return self._group

    @property
    def split(self) -> Optional[str]:
        return self._split

    @property
    def time(self) -> Optional[List]:
        return self._time

    @property
    def ignore(self) -> Optional[Iterable[str]]:
        return self._ignore

    @property
    def config_src(self) -> Union[str, Path, dict, None]:
        return self._config_src

    @config_src.setter
    def config_src(self, value: Union[str, Path, dict, None]):
        self._config_src = value

    @property
    def default_config(self) -> Optional[str]:
        return self._default_config

    def __init__(
        self,
        *table: Union[str, Path, pd.DataFrame],
        sample_weight: Optional[str] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None,
        classify: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
        regress: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
        group: Optional[str] = None,
        split: Optional[str] = None,
        time: Optional[str] = None,
        ignore: Optional[Iterable[str]] = None,
        config: Union[str, Path, dict, None] = None,
        default_config: Optional[str] = None,
        **_
    ):

        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)

        self._classify = classify
        self._regress = regress
        self._group = group
        self._time = time
        self._ignore = ignore
        self._config_src = config
        self._default_config = default_config
        self._target = None

    def update(self, src: Dict = None):
        super().update(src)
        if src:
            if self._classify is None:
                self._classify = src.get('classify')
            if self._regress is None:
                self._regress = src.get('regress')
            if self._group is None:
                self._group = src.get('group')
            if self._time is None:
                self._time = src.get('time')
            if self._ignore is None:
                self._ignore = src.get('ignore')
            if self._config_src is None:
                self._config_src = src.get('config')
            if self._default_config is None:
                self._default_config = src.get('default_config')

    def resolve(self):
        if self._group == '':
            self._group = None
        if self._config_src == '':
            self._config_src = None
        if self._default_config in (None, ''):
            self._default_config = 'full'
        self._ignore = set() if self._ignore is None else set(self._ignore)

        if self._classify is None:
            if self._regress is None:
                self._target = []
            else:
                self._target = [self._regress] if isinstance(self._regress, str) else list(self._regress)
            self._classify = False
        elif self._regress is None:
            self._target = [self._classify] if isinstance(self._classify, str) else list(self._classify)
            self._classify = True
        else:
            raise ValueError('At least one of `classify` and `regress` must be None.')

        if self._out is None:
            self._out = self._table[0]
            if isinstance(self._out, pd.DataFrame):
                raise ValueError('Output directory must be specified when passing a DataFrame.')
            self._out = self._out.parent / (self._out.stem + '_catabra_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self.out = io.make_path(self._out, absolute=True)

    def to_dict(self) -> Dict:
        dic = super().to_dict()
        dic.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            target=['<DataFrame>' if isinstance(tgt, pd.DataFrame) else tgt for tgt in self._target],
            classify=self._classify,
            group=self._group,
            ignore=self._ignore,
            config=self._config_src,
            default_config=self._default_config,
            time=self._time,
        ))
        return dic

