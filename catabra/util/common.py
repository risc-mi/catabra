from typing import Union, Optional, Iterable
import itertools


def fresh_name(name, lst: Iterable):
    """
    Create a fresh name based on `name`, i.e., a name that does not appear in `lst`.
    :param name: An arbitrary object. If a list, tuple or set, all elements of `name` are processed individually, and
    they are ensured to be distinct from each other.
    :param lst: A list-like structure.
    :return: If `name` does not appear in `lst`, `name` is returned as-is. Otherwise, a numeric suffix is added to the
    string representation of `name`.
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
    :param lst: The list.
    :param limit: The maximum number of displayed items.
    :param delim: The item delimiter.
    :param brackets: Whether to add brackets.
    :return: String representation of `lst`.
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
