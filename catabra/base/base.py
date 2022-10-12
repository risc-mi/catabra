import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional, Iterable, Dict

import numpy as np
import pandas as pd

from catabra.base import io, logging

class CaTabRaBase:

    @property
    def invocation_src(self) -> Dict:
        return self._invocation_src

    @abstractmethod
    def __call__(self):
        pass

    def __init__(
            self,
            invocation: Union[str, Path, dict, None] = None
    ):
        if isinstance(invocation, (str, Path)):
            self._invocation_src = io.load(invocation)
        if isinstance(invocation, dict):
            self._invocation_src = invocation
        else:
            self._invocation_src = {}

    def _resolve_output_dir(self) -> bool:
        if self._invocation.out.exists():
            if logging.prompt(f'Output folder "{self.invocation.out.as_posix()}" already exists. Delete?',
                              accepted=['y', 'n'], allow_headless=False) == 'y':
                if self._invocation.out.is_dir():
                    shutil.rmtree(self._invocation.out.as_posix())
                else:
                    self._invocation.out.unlink()
            else:
                logging.log('### Aborting')
                return False

        self._invocation.out.mkdir(parents=True)
        return True

    def _get_sample_weights(self, df) -> Optional[np.ndarray]:
        if self._invocation.sample_weight is None:
            return None
        elif self._invocation.sample_weight in df.columns:
            if df[self._invocation.sample_weight].dtype.kind not in 'fiub':
                raise ValueError(f'Column "{self._invocation.sample_weight}" must have numeric data type,'
                                 f' but found {df[self._invocation.sample_weight].dtype.name}.')
            logging.log(f'Weighting samples by column "{self._invocation.sample_weight}"')
            # ignore.add(sample_weight)
            sample_weights = df[self._invocation.sample_weight].values
            na_mask = np.isnan(sample_weights)
            if na_mask.any():
                sample_weights = sample_weights.copy()
                sample_weights[na_mask] = 1.
            return sample_weights
        else:
            raise ValueError(f'"{self._invocation.sample_weight}" is no column of the specified table.')

