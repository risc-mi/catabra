import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional, Iterable

import numpy as np
import pandas as pd

from catabra.base import io, logging


#
# class InvocationArgs:
#     table = None
#     group: str = None
#     split: str = None
#     sample_weight: str = None
#     ignore: list = None
#     out: str = None
#
#     def resolve(self, invocation: Dict[str, Any]):
#         for name, value in vars(self).items():
#             if value is None:
#                 self.__setattr__(name, invocation.get(name))


class CaTabRaBase:

    # invocation = InvocationArgs()

    @abstractmethod
    def __call__(self):
        pass

    def __init__(
            self,
            *table: Union[str, Path, pd.DataFrame],
            group: Optional[str] = None,
            split: Optional[str] = None,
            sample_weight: Optional[str] = None,
            ignore: Optional[Iterable[str]] = None,
            time: Optional[int] = None,
            out: Union[str, Path, None] = None,
            config: Union[str, Path, dict, None] = None,
            default_config: Optional[str] = None,
            jobs: Optional[int] = None,
            from_invocation: Union[str,Path, dict, None] = None
    ):
        if isinstance(from_invocation, (str, Path)):
            self._from_invocation = io.load(from_invocation)
        if isinstance(from_invocation, dict):
            self._from_invocation = from_invocation
            self._resolve_invocation_args()
        else:
            self._from_invocation = {}

        self._resolve_invocation_args(
            *table,
            group=group,
            split=split,
            sample_weight=sample_weight,
            ignore=ignore,
            time=time,
            out=out,
            config=config,
            default_config=default_config,
            jobs=jobs
        )

        self._validate_arguments()
        # sample_weights = self._get_sample_weights()

    def _resolve_invocation_args(
            self,
            *table: Union[str, Path, pd.DataFrame],
            group: Optional[str] = None,
            split: Optional[str] = None,
            sample_weight: Optional[str] = None,
            ignore: Optional[Iterable[str]] = None,
            time: Optional[int] = None,
            out: Union[str, Path, None] = None,
            config: Union[str, Path, dict, None] = None,
            default_config: Optional[str] = None,
            jobs: Optional[int] = None,
    ):
        if len(table) == 0:
            self._table = self._from_invocation.get('table') or []
            if '<DataFrame>' in table:
                raise ValueError('Invocations must not contain "<DataFrame>" tables.')
        else:
            self._table = table

        self._group = self._from_invocation.get('group') if group is None else group
        self._split = self._from_invocation.get('split') if split is None else split
        self._sample_weight = self._from_invocation.get('sample_weight') if sample_weight is None else sample_weight
        self._ignore = self._from_invocation.get('ignore') if ignore is None else ignore
        self._out = self._from_invocation.get('out') if out is None else out

        self._config = self._from_invocation.get('config', {}) if config is None else config
        self._default_config = self._from_invocation.get('default_config') if default_config is None else default_config
        self._time = self._from_invocation.get('time') if time is None else time
        self._jobs = self._from_invocation.get('jobs') if jobs is None else jobs

    def _validate_arguments(self):
        if self._group == '':
            self._group = None
        if self._split == '':
            self._split = None
        if self._sample_weight == '':
            self._sample_weight = None
        if self._config == '':
            self._config = None
        if self._default_config in (None, ''):
            self._default_config = 'full'

    def _resolve_output_dir(self) -> bool:
        if self._out.exists():
            if logging.prompt(f'Output folder "{self._out.as_posix()}" already exists. Delete?',
                              accepted=['y', 'n'], allow_headless=False) == 'y':
                if self._out.is_dir():
                    shutil.rmtree(self._out.as_posix())
                else:
                    self._out.unlink()
            else:
                logging.log('### Aborting')
                return False

        self._out.mkdir(parents=True)
        return True

    def _get_sample_weights(self, df) -> Optional[np.ndarray]:
        if self._sample_weight is None:
            return None
        elif self._sample_weight in df.columns:
            if df[self._sample_weight].dtype.kind not in 'fiub':
                raise ValueError(f'Column "{self._sample_weight}" must have numeric data type,'
                                 f' but found {df[self._sample_weight].dtype.name}.')
            logging.log(f'Weighting samples by column "{self._sample_weight}"')
            # ignore.add(sample_weight)
            sample_weights = df[self._sample_weight].values
            na_mask = np.isnan(sample_weights)
            if na_mask.any():
                sample_weights = sample_weights.copy()
                sample_weights[na_mask] = 1.
            return sample_weights
        else:
            raise ValueError(f'"{self._sample_weight}" is no column of the specified table.')

