#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import json
import pickle
from csv import Sniffer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import joblib
import pandas as pd

from catabra.core.paths import CaTabRaPaths


def make_path(p: Union[str, Path], absolute: bool = False) -> Path:
    """
    Convert a path-like object into a proper path object, i.e., an instance of class `Path`.

    Parameters
    ----------
    p: str | Path
        Path-like object. If an instance of `Path` and `absolute` is False, `p` is returned unchanged.
    absolute: bool, default=False
        Whether to make sure that the output is an absolute path. If False, the path may be relative.

    Returns
    -------
    Path
        Path object.
    """
    if not isinstance(p, Path):
        p = Path(p)
    if absolute:
        return p.absolute()
    return p


def read_df(fn: Union[str, Path], key: Union[str, Iterable[str]] = 'table') -> pd.DataFrame:
    """
    Read a DataFrame from a CSV, Excel, HDF5, Pickle or Parquet file. The file type is determined from the file
    extension of the given file.

    Parameters
    ----------
    fn: str | Path
        The file to read.
    key: str | Iterable[str], default='table'
        The key(s) in the HDF5 file, if `fn` is an HDF5 file. Defaults to "table". If an iterable, all keys are read and
        concatenated along the row axis.

    Returns
    -------
    DataFrame
        A DataFrame.
    """
    fn = make_path(fn)
    fmt, _ = _infer_file_format(fn.suffixes)
    if fmt == 'excel':
        return pd.read_excel(str(fn))
    elif fmt == 'csv':
        with open(fn, mode='rt') as f:
            dialect = Sniffer().sniff(f.read(8192))
            f.seek(0)
            df = pd.read_csv(f, index_col=0, dialect=dialect)
        return df
    elif fmt == 'hdf':
        if isinstance(key, str):
            return pd.read_hdf(fn, key=key)
        else:
            dfs = [pd.read_hdf(fn, key=k) for k in key]
            if dfs:
                return pd.concat(dfs, sort=False)
            else:
                raise RuntimeError('No keys to read.')
    elif fmt == 'pickle':
        return pd.read_pickle(str(fn))
    elif fmt == 'parquet':
        return pd.read_parquet(str(fn))
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def read_dfs(fn: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Read multiple DataFrames from a single file.

    * If an Excel file, all sheets are read and returned.
    * If an H5 file, all top-level keys are read and returned.
    * If any other file, the singleton dict `{"table": df}` is returned, where `df` is the single DataFrame contained
      in the file.

    Parameters
    ----------
    fn: str, Path
        The file to read.

    Returns
    -------
    str | DataFrame
        A dict mapping keys to DataFrames, possibly empty.
    """
    fn = make_path(fn)
    fmt, _ = _infer_file_format(fn.suffixes)
    if fmt == 'excel':
        return pd.read_excel(str(fn), sheet_name=None)
    elif fmt == 'hdf':
        with pd.HDFStore(str(fn), mode='r') as h5:
            # `k[1:]` to trim leading "/"; don't iterate over `h5.items()`; skip "/"
            out = {k[1:]: h5[k] for k in h5.keys() if len(k) > 1}
        return out
    else:
        return dict(table=read_df(fn))


def write_df(df: pd.DataFrame, fn: Union[str, Path], key: str = 'table', mode: str = 'w'):
    """
    Write a DataFrame to file. The file type is determined from the file extension of the given file.

    Parameters
    ----------
    df: DataFrame
        The DataFrame to write.
    fn: str | Path
        The target file name.
    key: str, default='table'
        The key in the HDF5 file, if `fn` is an HDF5 file. If None, `fn` may contain only one table.
    mode: str, default='w'
        The mode in which the HDF5 file shall be opened, if `fn` is an HDF5 file. Ignored otherwise.
    """
    fn = make_path(fn)
    fn.parent.mkdir(exist_ok=True, parents=True)
    fmt, compression = _infer_file_format(fn.suffixes)
    if fmt == 'excel':
        delta_cols = [c for c in df.columns if df[c].dtype.kind == 'm']
        if delta_cols:
            # convert Timedelta columns to string columns
            df = df.copy()
            for c in delta_cols:
                df[c] = df[c].astype('str')
        df.to_excel(str(fn))
    elif fmt == 'csv':
        df.to_csv(str(fn), sep=';')
    elif fmt == 'hdf':
        if any(df[c].dtype.name == 'category' for c in df.columns):
            df.to_hdf(fn, key, mode=mode, format='table', complevel=9)
        else:
            df.to_hdf(fn, key, mode=mode, complevel=9)
    elif fmt == 'pickle':
        df.to_pickle(str(fn))
    elif fmt == 'parquet':
        df.to_parquet(str(fn), compression=compression)
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def write_dfs(dfs: Dict[str, pd.DataFrame], fn: Union[str, Path], mode: str = 'w'):
    """
    Write a dict of DataFrames to file. The file type is determined from the file extension of the given file.
    Unless an Excel- or HDF5 file, `dfs` must be empty or a singleton.

    Parameters
    ----------
    dfs: dict
        The DataFrames to write. If empty and `mode` differs from "a", the file is deleted.
    fn: str | Path
        The target file name.
    mode: str, default='w'
        The mode in which the file shall be opened, if `fn` is an Excel- or HDF5 file. Ignored otherwise.
    """
    fn = make_path(fn)
    if not dfs:
        if mode != 'a' and fn.exists():
            fn.unlink()
        return

    fn.parent.mkdir(exist_ok=True, parents=True)
    fmt, _ = _infer_file_format(fn.suffixes)
    if fmt == 'excel':
        with pd.ExcelWriter(str(fn), mode=mode) as writer:
            for k, df in dfs.items():
                delta_cols = [c for c in df.columns if df[c].dtype.kind == 'm']
                if delta_cols:
                    # convert Timedelta columns to string columns
                    df = df.copy()
                    for c in delta_cols:
                        df[c] = df[c].astype('str')
                df.to_excel(writer, sheet_name=str(k))
    elif fmt == 'hdf':
        with pd.HDFStore(str(fn), mode=mode) as h5:
            for k, df in dfs.items():
                h5[str(k)] = df
    elif len(dfs) > 1:
        raise RuntimeError(f'Cannot write more than one DataFrame to a "{fn.suffix}" file.')
    else:
        write_df(list(dfs.values())[0], fn)


def load(fn: Union[str, Path]):
    """
    Load a Python object from disk. The object can be stored in JSON, Pickle or joblib format. The format is
    automatically determined based on the given file extension:

    * ".json" => JSON
    * ".pkl", ".pickle" => Pickle
    * ".joblib" => joblib

    Parameters
    ----------
    fn: str | Path
        The file to load.

    Returns
    -------
    Any
        The loaded object.
    """
    fn = make_path(fn)
    fmt, _ = _infer_file_format(fn.suffixes)
    if fmt == 'json':
        with open(fn, mode='rt') as f:
            return json.load(f)
    elif fmt == 'pickle':
        with open(fn, mode='rb') as f:
            return pickle.load(f)
    elif fmt == 'joblib':
        return joblib.load(fn.as_posix())
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def dump(obj, fn: Union[str, Path]):
    """
    Dump a Python object to disk, either as a JSON, Pickle or joblib file. The format is determined automatically based
    on the given file extension:

    * ".json" => JSON
    * ".pkl", ".pickle" => Pickle
    * ".joblib" => joblib

    Parameters
    ----------
    obj:
        The object to dump.
    fn: str | Path
        The file.

    Notes
    -----
    When dumping objects as JSON, calling `to_json()` beforehand might be necessary to ensure compliance with the JSON
    standard. joblib is preferred over Pickle, as it is more efficient if the object contains large Numpy arrays.

    """
    fn = make_path(fn)
    fmt, _ = _infer_file_format(fn.suffixes)
    if fmt == 'json':
        with open(fn, mode='wt') as f:
            json.dump(obj, f, indent=2)
    elif fmt == 'pickle':
        with open(fn, mode='wb') as f:
            pickle.dump(obj, f)
    elif fmt == 'joblib':
        joblib.dump(obj, fn.as_posix())
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def to_json(x):
    """
    Returns a JSON-compliant representation of the given object.

    Parameters
    ----------
    x:
        Arbitrary object.

    Returns
    -------
    Any
        Representation of `x` that can be serialized as JSON.
    """
    if isinstance(x, Path):
        return x.as_posix()
    elif isinstance(x, (pd.Timedelta, pd.Timestamp, type(pd.NaT))):
        return str(x)
    elif isinstance(x, (list, tuple, set)):
        return [to_json(y) for y in x]
    elif isinstance(x, dict):
        return {str(k): to_json(v) for k, v, in x.items()}
    elif hasattr(x, 'tolist'):
        return x.tolist()
    elif hasattr(x, 'to_dict'):
        return to_json(x.to_dict())
    elif x in (None, True, False) or isinstance(x, (str, int, float)):
        return x
    else:
        return str(x)


def convert_rows_to_str(d: [dict, pd.DataFrame], rowindex_to_convert: list,
                        inplace: bool = True, skip: list = []) -> Union[dict, pd.DataFrame]:
    """
    Converts rows (indexed via rowindex_to_convert) to str, mainly used for saving dataframes (to avoid missing values
    in .xlsx-files in case of e.g. timedelta datatype)

    Parameters
    ----------
    d: dict | DataFrame
        Single DataFrame or dictionary of dataframes
    rowindex_to_convert: list
        List of row indices (e.g., features), that should be converted to str
    inplace: bool, default=True
        Determines if changes will be made to input data or a deep-copy of it
    skip: list, default=[]
        List of column(s) that should not be converted to string

    Returns
    -------
    DataFrame | dict
        Modified (str-converted rows) single DataFrame or dictionary of DataFrames.
    """
    if not inplace:
        if isinstance(d, pd.DataFrame):
            d = d.copy()
        elif isinstance(d, dict):
            import copy
            d = copy.deepcopy(d)

    if isinstance(d, pd.DataFrame):
        d.loc[rowindex_to_convert, ~d.columns.isin(skip)] = d.loc[
            rowindex_to_convert, ~d.columns.isin(skip)].astype(str)
    elif isinstance(d, dict):
        for key_ in list(d.keys()):
            if isinstance(d[key_], pd.DataFrame):
                d[key_].loc[rowindex_to_convert, ~d[key_].columns.isin(skip)] = d[key_].loc[
                    rowindex_to_convert, ~d[key_].columns.isin(skip)].astype(str)
    return d


class CaTabRaLoader:
    """
    CaTabRaLoader for conveniently accessing artifacts generated by analyzing tables, like trained models, configs,
    encoders, etc.

    Parameters
    ----------
    path: str | Path
        Path to the CaTabRa directory.
    check_exists: bool, default=True
        Check whether the directory pointed to by `path` exists.
    """

    def __init__(self, path: Union[str, Path], check_exists: bool = True):
        self._path = make_path(path, absolute=True)
        if check_exists and not self._path.exists():
            raise ValueError(f'CaTabRa directory "{self._path.as_posix()}" does not exist.')

    @property
    def path(self) -> Path:
        return self._path

    def get_config(self) -> Optional[dict]:

        return self._load(CaTabRaPaths.Config)

    def get_invocation(self) -> Optional[dict]:
        return self._load(CaTabRaPaths.Invocation)

    def get_model_summary(self) -> Optional[dict]:
        return self._load(CaTabRaPaths.ModelSummary)

    def get_training_history(self) -> Optional[pd.DataFrame]:
        if (self._path / CaTabRaPaths.TrainingHistory).exists():
            return read_df(self._path / CaTabRaPaths.TrainingHistory)

    def get_encoder(self) -> Optional['Encoder']: # noqa F821
        if (self._path / CaTabRaPaths.Encoder).exists():
            from ..util.encoding import Encoder
            return Encoder.load(self._path / CaTabRaPaths.Encoder)

    def get_model(self) -> Optional['AutoMLBackend']: # noqa F821
        return self._load(CaTabRaPaths.Model)

    def get_ood(self) -> Optional['OODDetector']: # noqa F821
        return self._load(CaTabRaPaths.OODModel)

    def get_fitted_ensemble(self, from_model: bool = False) -> Optional['FittedEnsemble']: # noqa F821
        """
        Get the trained prediction model as a FittedEnsemble object.

        Parameters
        ----------
        from_model: bool, default=False
            Whether to convert a plain model of type AutoMLBackend into a FittedEnsemble object, if such an object does
            not exist in the directory.
        """
        if (self._path / CaTabRaPaths.FittedEnsemble).exists():
            from ..automl.fitted_ensemble import FittedEnsemble
            return FittedEnsemble.load(self._path / CaTabRaPaths.FittedEnsemble)
        elif from_model:
            model = self.get_model()
            if hasattr(model, 'fitted_ensemble'):
                return model.fitted_ensemble()

    def get_model_or_fitted_ensemble(self) -> Union['AutoMLBackend', 'FittedEnsemble', None]: # noqa F821
        return self.get_model() or self.get_fitted_ensemble()

    def get_explainer(self, explainer: Optional[str] = None, fitted_ensemble: Optional['FittedEnsemble'] = None) \
            -> Optional['EnsembleExplainer']: # noqa F821
        """
        Get the explainer object.

        Parameters
        ----------
        explainer: str, optional
            Name of the explainer to load. If None, the first explainer specified in config param "explainer" is loaded.
        fitted_ensemble: FittedEnsemble
            Pre-loaded FittedEnsemble object. If None, method `get_fitted_ensemble()` is used for loading it.
        """
        config = self.get_config() or {}
        if explainer is None:
            explainer = config.get('explainer') or []
            if isinstance(explainer, (list, set, tuple)):
                if explainer:
                    explainer = explainer[0]
                else:
                    return None

        if (self._path / explainer / 'params.joblib').exists():
            params = load(self._path / explainer / 'params.joblib')
        else:
            params = None
        if fitted_ensemble is None:
            fitted_ensemble = self.get_fitted_ensemble(from_model=True)
        if fitted_ensemble is not None:
            from ..explanation import EnsembleExplainer
            try:
                return EnsembleExplainer.get(explainer, config=config, ensemble=fitted_ensemble, params=params)
            except:     # noqa
                if params is None:
                    # exception may be caused by missing `params`
                    # silently return None to be consistent with original behavior
                    return None
                else:
                    raise

    def get_train_data(self) -> Optional[pd.DataFrame]:
        """
        Get the training data copied into the directory, "train_data.h5". In contrast to `get_table()`, this is only
        the data actually used for training.
        """
        if (self._path / CaTabRaPaths.TrainData).exists():
            return read_df(self._path / CaTabRaPaths.TrainData)

    def get_table(self, keep_singleton: bool = False) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
        """
        Get the table(s) originally passed to `analyze()`, if they still reside in their original location.

        Parameters
        ----------
        keep_singleton: bool, default=False
            Whether to keep singleton lists. If False, a single DataFrame is returned in that case.
        """
        table = (self.get_invocation() or {}).get('table')
        if table is not None:
            if not isinstance(table, list):
                table = [table]
            table = [make_path(t) for t in table]
            if all(t.exists() for t in table):
                table = [read_df(t) for t in table]
                if len(table) == 1 and not keep_singleton:
                    return table[0]
                return table

    def _load(self, name: str):
        if (self._path / name).exists():
            return load(self._path / name)


def _infer_file_format(suffixes: list) -> (Optional[str], Optional[str]):
    suffixes = [suffix.lower() for suffix in suffixes]
    compression = None
    for suffix in suffixes[::-1]:
        if suffix in ('.gzip', '.zip', '.xy', '.bz2', '.snappy', '.brotli'):
            compression = suffix[1:]
        elif suffix in ('.pickle', '.pkl'):
            return 'pickle', compression
        elif suffix == '.joblib':
            return 'joblib', compression
        elif suffix == '.json':
            return 'json', compression
        elif suffix in ('.xls', '.xlsx'):
            return 'excel', compression
        elif suffix in ('.h5', '.hdf'):
            return 'hdf', compression
        elif suffix == '.csv':
            return 'csv', compression
        elif suffix.startswith('.parquet'):
            return 'parquet', compression
        else:
            return None, None
