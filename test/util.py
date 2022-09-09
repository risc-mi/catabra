import pandas as pd


def profile(fun, *args, _prefix: str = '', _fun_name=None, _include_kwargs: list = None, **kwargs):
    if _fun_name is None:
        _fun_name = fun.__name__
    _include_kwargs = _include_kwargs or []
    _include_kwargs = [(k, kwargs[k]) for k in _include_kwargs if k in kwargs]
    if _include_kwargs:
        _fun_name += '('
        for k, v in _include_kwargs:
            _fun_name += k + '=' + repr(v) + ', '
        _fun_name = _fun_name[:-2] + ')'
    tic = pd.Timestamp.now()
    print(_prefix + '> ' + _fun_name)
    res = fun(*args, **kwargs)
    toc = pd.Timestamp.now()
    print(_prefix + '< ' + _fun_name + ': ' + str(toc - tic))
    return res
