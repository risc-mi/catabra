from typing import Optional, Tuple
import numpy as np
import pandas as pd


def make_title(name: Optional[str], title: Optional[str], sep: str = '\n') -> Optional[str]:
    if title is None:
        return name
    elif name is None:
        return title
    else:
        return title + sep + name


def convert_timedelta(x: np.ndarray) -> Tuple[pd.Timedelta, str]:
    mean = np.abs(x).mean()
    unit = pd.Timedelta(365.2525, unit='d')
    if mean > unit:
        uom = 'y'
    else:
        unit = pd.Timedelta(1, unit='d')
        if mean > unit:
            uom = 'd'
        else:
            unit = pd.Timedelta(1, unit='h')
            if mean > unit:
                uom = 'h'
            else:
                unit = pd.Timedelta(1, unit='m')
                if mean > unit:
                    uom = 'm'
                else:
                    unit = pd.Timedelta(1, unit='s')
                    uom = 's'
    return unit, uom
