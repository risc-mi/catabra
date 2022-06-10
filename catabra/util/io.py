from typing import Union, Optional, Iterable, Dict, List
from pathlib import Path
import json
import pickle
import joblib
import pandas as pd
from csv import Sniffer


def make_path(p: Union[str, Path], absolute: bool = False) -> Path:
    """
    Convert a path-like object into a proper path object, i.e., an instance of class `Path`.
    :param p: Path-like object. If an instance of `Path` and `absolute` is False, `p` is returned unchanged.
    :param absolute: Whether to make sure that the output is an absolute path. If False, the path may be relative.
    :return: Path object.
    """
    if not isinstance(p, Path):
        p = Path(p)
    if absolute:
        return p.absolute()
    return p


def read_df(fn: Union[str, Path], key: Union[str, Iterable[str]] = 'table') -> pd.DataFrame:
    """
    Read a DataFrame from a CSV, Excel or HDF5 file. The file type is determined from the file extension of the given
    file.
    :param fn: The file to read.
    :param key: The key(s) in the HDF5 file, if `fn` is an HDF5 file. Defaults to "table". If an iterable, all
    keys are read an concatenated along the row axis.
    :return: A DataFrame.
    """
    fn = make_path(fn)
    if fn.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(str(fn))
    elif fn.suffix.lower() == '.csv':
        with open(fn, mode='rt') as f:
            dialect = Sniffer().sniff(f.read(8192))
            f.seek(0)
            df = pd.read_csv(f, index_col=0, dialect=dialect)
        return df
    elif fn.suffix.lower() in ('.h5', '.hdf'):
        if isinstance(key, str):
            return pd.read_hdf(fn, key=key)
        else:
            dfs = [pd.read_hdf(fn, key=k) for k in key]
            if dfs:
                return pd.concat(dfs, sort=False)
            else:
                raise RuntimeError('No keys to read.')
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def write_df(df: pd.DataFrame, fn: Union[str, Path], key: str = 'table', mode: str = 'w'):
    """
    Write a DataFrame to file. The file type is determined from the file extension of the given file.
    :param df: The DataFrame to write.
    :param fn: The target file name.
    :param key: The key in the HDF5 file, if `fn` is an HDF5 file. If None, `fn` may contain only one table.
    :param mode: The mode in which the HDF5 file shall be opened, if `fn` is an HDF5 file. Ignored otherwise.
    """
    fn = make_path(fn)
    fn.parent.mkdir(exist_ok=True, parents=True)
    if fn.suffix.lower() in ('.xlsx', '.xls'):
        delta_cols = [c for c in df.columns if df[c].dtype.kind == 'm']
        if delta_cols:
            # convert Timedelta columns to string columns
            df = df.copy()
            for c in delta_cols:
                df[c] = df[c].astype('str')
        df.to_excel(str(fn))
    elif fn.suffix.lower() == '.csv':
        df.to_csv(str(fn), sep=';')
    elif fn.suffix.lower() in ('.h5', '.hdf'):
        if any(df[c].dtype.name == 'category' for c in df.columns):
            df.to_hdf(fn, key, mode=mode, format='table', complevel=9)
        else:
            df.to_hdf(fn, key, mode=mode, complevel=9)
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def write_dfs(dfs: Dict[str, pd.DataFrame], fn: Union[str, Path], mode: str = 'w'):
    """
    Write a dict of DataFrames to file. The file type is determined from the file extension of the given file.
    If a CSV file, `dfs` must be empty or a singleton.
    :param dfs: The DataFrames to write. If empty and `mode` differs from "a", the file is deleted.
    :param fn: The target file name.
    :param mode: The mode in which the file shall be opened, if `fn` is an Excel- or HDF5 file. Ignored otherwise.
    """
    fn = make_path(fn)
    if not dfs:
        if mode != 'a' and fn.exists():
            fn.unlink()
        return

    fn.parent.mkdir(exist_ok=True, parents=True)
    if fn.suffix.lower() in ('.xlsx', '.xls'):
        with pd.ExcelWriter(str(fn), mode=mode) as writer:
            for k, df in dfs.items():
                delta_cols = [c for c in df.columns if df[c].dtype.kind == 'm']
                if delta_cols:
                    # convert Timedelta columns to string columns
                    df = df.copy()
                    for c in delta_cols:
                        df[c] = df[c].astype('str')
                df.to_excel(writer, sheet_name=k)
    elif fn.suffix.lower() == '.csv':
        if len(dfs) > 1:
            raise RuntimeError('Cannot write more than one DataFrame to a CSV file.')
        else:
            write_df(list(dfs.values())[0], fn)
    elif fn.suffix.lower() in ('.h5', '.hdf'):
        with pd.HDFStore(str(fn), mode=mode) as h5:
            for k, df in dfs.items():
                h5[k] = df
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def load(fn: Union[str, Path]):
    """
    Load a Python object from disk. The object can be stored in JSON, Pickle or joblib format. The format is
    automatically determined based on the given file extension:
    * ".json" => JSON
    * ".pkl", ".pickle" => Pickle
    * ".joblib" => joblib
    :param fn: The file to load.
    :return: The loaded object.
    """
    fn = make_path(fn)
    if fn.suffix.lower() == '.json':
        with open(fn, mode='rt') as f:
            return json.load(f)
    elif fn.suffix.lower() in ('.pkl', '.pickle'):
        with open(fn, mode='rb') as f:
            return pickle.load(f)
    elif fn.suffix.lower() == '.joblib':
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

    When dumping objects as JSON, calling `to_json()` beforehand might be necessary to ensure compliance with the
    JSON standard.
    joblib is preferred over Pickle, as it is more efficient if the object contains large Numpy arrays.

    :param obj: The object to dump.
    :param fn: The file.
    """
    fn = make_path(fn)
    if fn.suffix.lower() == '.json':
        with open(fn, mode='wt') as f:
            json.dump(obj, f, indent=2)
    elif fn.suffix.lower() in ('.pkl', '.pickle'):
        with open(fn, mode='wb') as f:
            pickle.dump(obj, f)
    elif fn.suffix.lower() == '.joblib':
        joblib.dump(obj, fn.as_posix())
    else:
        raise RuntimeError(f'Unknown file format: "{fn.suffix}".')


def to_json(x):
    """
    Returns a JSON-compliant representation of the given object.
    :param x: Arbitrary object.
    :return: Representation of `x` that can be serialized as JSON.
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
    else:
        return x


def convert_rows_to_str(d: [dict, pd.DataFrame], rowindex_to_convert: list,
                        inplace: bool = True, skip: list = []) -> Union[dict, pd.DataFrame]:
    """
    Converts rows (indexed via rowindex_to_convert) to str, mainly used for saving dataframes (to avoid missing values
    in .xlsx-files in case of e.g. timedelta datatype)
    :param d: Single DataFrame or dictionary of dataframes
    :param rowindex_to_convert: List of row indices (e.g., features), that should be converted to str
    :param inplace: Determines if changes will be made to input data or a deep-copy of it
    :param skip: List of column(s) that should not be converted to string
    :return: Modified (str-converted rows) single DataFrame or dictionary of dataframes
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

    def __init__(self, path: Union[str, Path], check_exists: bool = True):
        """
        CaTabRaLoader for conveniently accessing artifacts generated by analyzing tables, like trained models, configs,
        encoders, etc.
        :param path: Path to the CaTabRa directory.
        :param check_exists: Check whether the directory pointed to by `path` exists.
        """
        self._path = make_path(path, absolute=True)
        if check_exists and not self._path.exists():
            raise ValueError(f'CaTabRa directory "{self._path.as_posix()}" does not exist.')

    @property
    def path(self) -> Path:
        return self._path

    def get_config(self) -> Optional[dict]:
        return self._load('config.json')

    def get_invocation(self) -> Optional[dict]:
        return self._load('invocation.json')

    def get_model_summary(self) -> Optional[dict]:
        return self._load('model_summary.json')

    def get_training_history(self) -> Optional[pd.DataFrame]:
        if (self._path / 'training_history.xlsx').exists():
            return read_df(self._path / 'training_history.xlsx')

    def get_encoder(self) -> Optional['Encoder']:
        if (self._path / 'encoder.json').exists():
            from .encoding import Encoder
            return Encoder.load(self._path / 'encoder.json')

    def get_model(self) -> Optional['AutoMLBackend']:
        return self._load('model.joblib')

    def get_fitted_ensemble(self, from_model: bool = False) -> Optional['FittedEnsemble']:
        """
        Get the trained prediction model as a FittedEnsemble object.
        :param from_model: Whether to convert a plain model of type AutoMLBackend into a FittedEnsemble object, if
        such an object does not exist in the directory.
        """
        if (self._path / 'fitted_ensemble.joblib').exists():
            from ..automl.fitted_ensemble import FittedEnsemble
            return FittedEnsemble.load(self._path / 'fitted_ensemble.joblib')
        elif from_model:
            model = self.get_model()
            if hasattr(model, 'fitted_ensemble'):
                return model.fitted_ensemble()

    def get_model_or_fitted_ensemble(self) -> Union['AutoMLBackend', 'FittedEnsemble', None]:
        return self.get_model() or self.get_fitted_ensemble()

    def get_explainer(self, fitted_ensemble: Optional['FittedEnsemble'] = None) -> Optional['EnsembleExplainer']:
        """
        Get the explainer object.
        :param fitted_ensemble: Pre-loaded FittedEnsemble object. If None, method `get_fitted_ensemble()` is used for
        loading it.
        """
        explainer = (self.get_config() or {}).get('explainer')
        if explainer is not None and (self._path / explainer / 'params.joblib').exists():
            params = load(self._path / explainer / 'params.joblib')
            if fitted_ensemble is None:
                fitted_ensemble = self.get_fitted_ensemble(from_model=True)
            if fitted_ensemble is not None:
                from ..explanation import EnsembleExplainer
                return EnsembleExplainer.get(explainer, ensemble=fitted_ensemble, params=params)

    def get_train_data(self) -> Optional[pd.DataFrame]:
        """
        Get the training data copied into the directory, "train_data.h5". In contrast to `get_table()`, this is only
        the data actually used for training.
        """
        if (self._path / 'train_data.h5').exists():
            return read_df(self._path / 'train_data.h5')

    def get_table(self, keep_singleton: bool = False) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
        """
        Get the table(s) originally passed to `analyze()`, if they still reside in their original location.
        :param keep_singleton: Whether to keep singleton lists. If False, a single DataFrame is returned in that case.
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
