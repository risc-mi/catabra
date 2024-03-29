#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import itertools
from typing import Iterable, Optional, Union


def fresh_name(name, lst: Iterable):
    """
    Create a fresh name based on `name`, i.e., a name that does not appear in `lst`.

    Parameters
    ----------
    name:
        An arbitrary object. If a list, tuple or set, all elements of `name` are processed individually, an they are
        ensured to be distinct from each other.
    lst: Iterable
        A list-like structure.

    Returns
    -------
    Any
        If `name` does not appear in `lst`, `name` is returned as-is. Otherwise, a numeric suffix is added to the string
        representation of `name`.
    """
    if isinstance(name, (list, tuple, set)):
        cstr = type(name)
        proc = []
        for n in name:
            proc.append(fresh_name(n, itertools.chain(lst, proc)))
        return proc if cstr is list else cstr(proc)
    elif name in lst:
        if not isinstance(name, str):
            name = str(name)
        len_n = len(name)
        y = -1
        for m in lst:
            if isinstance(m, str) and m.startswith(name):
                try:
                    x = int(m[len_n:])
                except ValueError:
                    pass
                else:
                    if x > y:
                        y = x
        return name + str(y + 1)
    else:
        return name


def repr_list(lst: Union[list, tuple], limit: Optional[int] = 50, delim: str = ', ', brackets: bool = True) -> str:
    """
    Return a string representation of some list, limiting the displayed items to a certain number.

    Parameters
    ----------
    lst: list | tuple
        The list.
    limit: int, default=50
        The maximum number of displayed items.
    delim: str, default=', '
        The item delimiter.
    brackets: bool, default=True
        Whether to add brackets.

    Returns
    -------
    str
        String representation of `lst`.
    """
    if limit is not None and len(lst) > limit:
        ell = '...'
        if limit == 0:
            out = ell
        elif limit == 1:
            out = repr(lst[0]) + delim + ell
        else:
            out = delim.join([repr(x) for x in lst[:limit // 2]]) + delim + ell + delim \
                  + delim.join([repr(x) for x in lst[-limit // 2:]])
    else:
        out = delim.join([repr(x) for x in lst])
    return '[' + out + ']' if brackets else out


def repr_timedelta(delta, subsecond_resolution: int = 0) -> str:
    """
    Return a string representation of some time delta.
    Minutes and seconds are always displayed, hours and days only if needed. Format is "d days hh:mm:ss".

    Parameters
    ----------
    delta:
        Time delta to represent, either a float or an object with a `total_seconds()` method (e.g., a pandas Timedelta
        instance). Floats are assumed to be given in seconds.
    subsecond_resolution: int, default=0
        The subsecond resolution to display, i.e., number of decimal places.

    Returns
    -------
    str
        String representation of `delta`.
    """
    assert subsecond_resolution >= 0
    if hasattr(delta, 'total_seconds'):
        delta = delta.total_seconds()
    out = ''
    if delta < 0:
        delta = -delta
        out = '-'
    if delta >= 3600:
        hours, delta = divmod(delta, 3600)
        hours = int(hours)
        if hours >= 24:
            days, hours = divmod(hours, 24)
            if days == 1:
                out += '1 day '
            else:
                out += '{:d} days '.format(days)
        out += '{:02d}:'.format(hours)
    minutes, delta = divmod(delta, 60)
    minutes = int(minutes)
    out += '{:02d}:'.format(minutes)
    sec = ('{:.' + str(subsecond_resolution) + 'f}').format(delta)
    comma = sec.find('.')
    if comma < 0:
        comma = len(sec)
    if comma < 2:
        out += '0' * (2 - comma)
    out += sec
    return out


def get_versions() -> dict:
    import sys

    from ..__version__ import __version__ as catabra_version

    out = dict(
        python='.'.join([str(sys.version_info.major), str(sys.version_info.minor), str(sys.version_info.micro)]),
        catabra=catabra_version
    )

    try:
        import numpy
        out['numpy'] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        out['pandas'] = pandas.__version__
    except ImportError:
        pass

    try:
        import sklearn
        out['scikit-learn'] = sklearn.__version__
    except ImportError:
        pass

    return out


def save_versions(versions: dict, file: str):
    with open(file, mode='wt') as f:
        for pkg, v in versions.items():
            f.write(f'{pkg}=={v}\n')
