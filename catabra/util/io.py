from typing import Union, Iterable, Dict
from pathlib import Path
import pandas as pd
from csv import Sniffer


def read_df(fn: Union[str, Path], key: Union[str, Iterable[str]] = 'table') -> pd.DataFrame:
    """
    Read a DataFrame from a CSV, Excel or HDF5 file. The file type is determined from the file extension of the given
    file.
    :param fn: The file to read.
    :param key: The key(s) in the HDF5 file, if `fn` is an HDF5 file. Defaults to "table". If an iterable, all
    keys are read an concatenated along the row axis.
    :return: A DataFrame.
    """
    if isinstance(fn, str):
        fn = Path(fn)
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
    if not isinstance(fn, Path):
        fn = Path(fn)
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
    if not isinstance(fn, Path):
        fn = Path(fn)
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
