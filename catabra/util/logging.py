#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
import sys
import time
import traceback
from typing import List, Optional

from catabra.util.common import repr_timedelta


def log(*msg, **kwargs):
    print('[CaTabRa]', *msg, flush=True, **kwargs)


def err(*msg, **kwargs):
    print('[CaTabRa error]', *msg, flush=True, file=sys.stderr, **kwargs)


def warn(*msg, **kwargs):
    print('[CaTabRa warning]', *msg, flush=True, file=sys.stderr, **kwargs)


def prompt(msg: str, accepted: Optional[List[str]] = None, allow_headless: bool = True) -> str:
    """
    Prompt the user for input.
    :param msg: The message to be printed.
    :param accepted: List of accepted inputs. Must be lower-case. If None, all inputs are accepted.
    :param allow_headless: What to do in headless mode. If True, the first element in `accepted` is returned if
    `accepted` is a list and "" is returned if `accepted` is None. If False, a RunTimeError is raised.
    :return: The input of the user, an element of `accepted` if `accepted` is a list, or arbitrary if `accepted` is
    None.
    """
    if Headless.headless():
        if allow_headless:
            assert not any(c.isupper() for a in accepted for c in a)
            if isinstance(accepted, list):
                return accepted[0]
            else:
                return ''
        else:
            raise RuntimeError(f'Input prompt "{msg}" in headless mode.')
    else:
        if isinstance(accepted, list):
            assert accepted
            assert not any(c.isupper() for a in accepted for c in a)
            msg += f' [{"/".join(accepted)}] '
        else:
            msg += ' '
        res = input(msg).lower()
        if accepted is None or res in accepted:
            return res
        else:
            msg = f'Not understood; please type in one of {", ".join(accepted)}: '
            while True:
                res = input(msg).lower()
                if res in accepted:
                    return res


def progress_bar(iterable, desc: Optional[str] = None, total: Optional[int] = None, disable: bool = False,
                 meter_width: int = 40):
    """
    Show a simple progress bar when iterating over a given iterable. This works similar to package `tqdm`, but in
    contrast to `tqdm` also works when mirroring messages to a file.
    :param iterable: The iterable.
    :param desc: Description to add to the beginning of the progress bar, optional.
    :param total: Total number of elements in `iterable` if `iterable` does not implement the `__len__()` method.
    :param disable: Whether to disable the progress bar. If True, the behavior is equivalent to not calling this
    function at all.
    :param meter_width: The width of the meter, in characters. Should not be too long to make the whole progress bar
    fit into a single line. Might have to be decreased if `desc` is a long text.
    """
    if disable:
        for obj in iterable:
            yield obj
    else:
        try:
            total = len(iterable)
        except AttributeError:
            pass

        if desc is None:
            desc = ''
        else:
            desc = desc + ': '

        state = dict(prev_len=0)

        def _get_speed(_i: int, _elapsed: float) -> str:
            if _i == 0:
                return '?it/s'
            if _i >= _elapsed:
                return '{:.2f}it/s'.format(_i / _elapsed)
            else:
                return '{:.2f}s/it'.format(_elapsed / _i)

        def _print(text: str, nl: bool = True, r: bool = False):
            prev_len = len(text)
            pl = state['prev_len']
            if r:
                if prev_len < pl:
                    text += ' ' * (pl - prev_len)
                text = '\r' + text
                state['prev_len'] = 0 if nl else prev_len
            elif nl:
                state['prev_len'] = 0
            else:
                state['prev_len'] = pl + prev_len
            print(text, end='\n' if nl else '')

        tic = time.time()
        i = 0
        if total is None:
            _print(desc + '0 [?it/s]', nl=False)
            for i, obj in enumerate(iterable, start=1):
                yield obj
                elapsed = time.time() - tic
                _print(desc + '{} [{}, {}]'.format(i, repr_timedelta(elapsed), _get_speed(i, elapsed)),
                       nl=False, r=True)

            _print(desc + '{} [{}, 100%]'.format(i, repr_timedelta(time.time() - tic)), r=True)
        elif total <= 0:
            print(desc + '100%|' + '#' * meter_width + '| 0/0 [00:00<00:00, ?it/s]')
        else:
            _print(desc + '  0%|' + ' ' * meter_width + '| 0/{} [00:00<??:??, ?it/s]'.format(total), nl=False)
            for i, obj in enumerate(iterable, start=1):
                yield obj
                elapsed = time.time() - tic
                m = int(meter_width * i / total)
                speed = _get_speed(i, elapsed)
                _print(
                    desc + '{:3d}%|'.format(round(i * 100 / total)) + '#' * m + ' ' * (meter_width - m) +
                    '| {}/{} [{}<{}, {}]'.format(i, total, repr_timedelta(elapsed),
                                                 repr_timedelta((total - i) * elapsed / i), speed),
                    nl=(i == total),
                    r=True
                )


class Headless:
    __VALUE = False

    @staticmethod
    def headless() -> bool:
        return Headless.__VALUE

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._active = False
        self._headless = None

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self):
        self.activate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()

    def activate(self):
        if not self._active:
            self._headless = Headless.__VALUE
            Headless.__VALUE = self._enabled
            self._active = True

    def deactivate(self):
        if self._active:
            Headless.__VALUE = self._headless
            self._active = False


class LogHide:

    def __init__(self, enabled: bool = True):
        """
        creates a new log_hide, use in a with statement to hide console output (stdout/stderr)
        example:
            with LogHide(verbosity < spam_me):
                call_verbose_package()
        :param enabled: if False, the instance turns into a dummy object not actually doing anything, this is useful for
        optionally hiding output
        """
        self._enabled = enabled
        self._active = False

    def __enter__(self):
        if self._enabled:
            self._activate(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._activate(False)

    def _activate(self, active: bool):
        if self._active != active:
            if active:
                self._stderr = sys.stderr
                self._stdout = sys.stdout
                self.__null = open(os.devnull, 'w')
                sys.stdout = sys.stderr = self.__null
            else:
                sys.stderr = self._stderr
                sys.stdout = self._stdout
                self.__null.close()
            self._active = active


class LogMirror:
    """
    used to temporary mirror both stderr and stdout to a log file
    Based on http://www.tentech.ca/2011/05/stream-tee-in-python-saving-stdout-to-file-while-keeping-the-console-alive/
    Based on https://gist.github.com/327585 by Anand Kunal
    example:
        with LogMirror("log.txt"):
            log("writing to log.txt and the console")
            err("works with errors as well")
            warn("and in case you need warnings")
            print("no need to use the custom log functions")
    """
    def __init__(self, log_path: str, mode: str = 'w'):
        self._log_path = log_path
        self._mode = mode

    def __enter__(self):
        self._file = open(self._log_path, mode=self._mode)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._tee(self._stdout, self._file)
        sys.stderr = self._tee(self._stderr, self._file)
        return self

    def __exit__(self, type, value, tb):
        if tb is not None:
            traceback.print_exc(file=self._file)
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()

    def print_to_file(self, msg: str):
        self._file.write(msg + '\n')

    class _tee:
        def __init__(self, stream1, stream2):
            self._stream1 = stream1
            self._stream2 = stream2
            self.__missing_method_name = None  # Hack!

        def __getattribute__(self, name):
            return object.__getattribute__(self, name)

        def __getattr__(self, name):
            self.__missing_method_name = name  # Could also be a property
            return getattr(self, '__methodmissing__')

        def __methodmissing__(self, *args, **kwargs):
            # Emit method call to stream 2
            try:
                callable2 = getattr(self._stream2, self.__missing_method_name)
                callable2(*args, **kwargs)
            except:     # noqa
                pass

            # Emit method call to stream 1
            try:
                callable1 = getattr(self._stream1, self.__missing_method_name)
                return callable1(*args, **kwargs)
            except:     # noqa
                pass
