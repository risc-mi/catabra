#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from catabra.util import io, logging


class Invocation(ABC):

    @property
    def start(self) -> pd.Timestamp:
        return self._start

    @property
    def table(self) -> Tuple[Union[str, Path, pd.DataFrame], ...]:
        return self._table

    @property
    def split(self) -> str:
        return self._split

    @property
    def sample_weight(self) -> Optional[str]:
        return self._sample_weight

    @property
    def out(self) -> Union[str, Path]:
        return self._out

    @out.setter
    def out(self, value: str):
        self._out = value

    @property
    def jobs(self) -> int:
        return self._jobs

    def __init__(
        self,
        *table: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        sample_weight: Optional[str] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None
    ):
        self._start = pd.Timestamp.now()
        self._table = table
        self._split = split
        self._sample_weight = sample_weight
        self._out = out
        self._jobs = jobs

    def update(self, src: Dict):
        if src:
            if len(self._table) == 0:
                self._table = src.get('table') or []
                if '<DataFrame>' in self._table:
                    raise ValueError('Invocations must not contain "<DataFrame>" tables.')

            if self._split is None:
                self._split = src.get('split')
            if self._sample_weight is None:
                self._sample_weight = src.get('sample_weight')

            if self._out is None:
                self._out = src.get('out')
            if self._jobs is None:
                self._jobs = src.get('jobs')

    def resolve(self):
        if self._split == '':
            self._split = None
        if self._sample_weight == '':
            self._sample_weight = None

        self._table = [io.make_path(tbl, absolute=True) if isinstance(tbl, (str, Path)) else tbl for tbl in self._table]
        if self.requires_table() and len(self._table) == 0:
            raise ValueError('No table specified.')

    def resolve_output_dir(self, prompt: str) -> bool:
        if self._out.exists():
            if logging.prompt(prompt, accepted=['y', 'n'], allow_headless=False) == 'y':
                if self._out.is_dir():
                    shutil.rmtree(self._out.as_posix())
                else:
                    self._out.unlink()
            else:
                return False

        self._out.mkdir(parents=True)
        return True

    def get_sample_weights(self, df) -> Optional[np.ndarray]:
        if self._sample_weight is None or df is None:
            return None
        elif self._sample_weight in df.columns:
            if df[self._sample_weight].dtype.kind not in 'fiub':
                raise ValueError(f'Column "{self._sample_weight}" must have numeric data type,'
                                 f' but found {df[self._sample_weight].dtype.name}.')
            logging.log(f'Weighting samples by column "{self._sample_weight}"')
            sample_weights = df[self._sample_weight].values
            na_mask = np.isnan(sample_weights)
            if na_mask.any():
                sample_weights = sample_weights.copy()
                sample_weights[na_mask] = 1.
            return sample_weights
        else:
            raise ValueError(f'"{self._sample_weight}" is no column of the specified table.')

    def to_dict(self) -> dict:
        return dict(
            split=self._split,
            sample_weight=self._sample_weight,
            out=self._out,
            jobs=self._jobs,
            timestamp=self._start
        )

    @staticmethod
    @abstractmethod
    def requires_table() -> bool:
        pass


class CaTabRaBase(ABC):

    @property
    @abstractmethod
    def invocation_class(self) -> Type[Invocation]:
        pass

    def __init__(
            self,
            invocation: Union[str, Path, dict, None] = None
    ):
        if isinstance(invocation, (str, Path)):
            self._invocation_src = io.load(invocation)
        elif isinstance(invocation, dict):
            self._invocation_src = invocation
        else:
            self._invocation_src = {}

    def __call__(self, *table, **kwargs):
        self._invocation = self.invocation_class(*table, **kwargs)
        self._invocation.update(self._invocation_src)
        self._invocation.resolve()
        self._call()

    @abstractmethod
    def _call(self):
        pass
