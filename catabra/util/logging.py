from typing import List
import os
import sys
import traceback


def log(*msg, **kwargs):
    print(*msg, flush=True, **kwargs)


def err(*msg, **kwargs):
    print('ERROR:', *msg, flush=True, file=sys.stderr, **kwargs)


def warn(*msg, **kwargs):
    print('WARNING:', *msg, flush=True, file=sys.stderr, **kwargs)


def info(*msg, **kwargs):
    print('INFO:', *msg, flush=True, file=sys.stderr, **kwargs)


class LogHide:

    def __init__(self, enabled: bool = True):
        """
        creates a new log_hide, use in a with statement to hide console output (stdout/stderr)
        example:
            with LogHide(verbosity < spam_me):
                call_verbose_package()
        :param enabled: if False, the instance turns into a dummy object not actually doing anything, this is useful for optionally hiding output
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
    def __init__(self, log_path: str):
        self._log_path = log_path

    def __enter__(self):
        self._file = open(self._log_path, 'w')
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
            except:
                pass

            # Emit method call to stream 1
            try:
                callable1 = getattr(self._stream1, self.__missing_method_name)
                return callable1(*args, **kwargs)
            except:
                pass