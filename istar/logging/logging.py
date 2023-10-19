import inspect
import logging
from functools import wraps
from logging import (  # noqa: F401
    DEBUG,
    ERROR,
    INFO,
    WARNING,
)
from typing import List, Any
import warnings

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)

_PROGRESSBARS: List[tqdm] = []


def _refresh_progressbars():
    # pylint: disable=protected-access
    for i, pbar in enumerate(reversed(_PROGRESSBARS)):
        if pbar._tqdm_instance.pos != i:
            pbar._tqdm_instance.clear()
            pbar._tqdm_instance.pos = i
            pbar._tqdm_instance.refresh()


@wraps(LOGGER.log)
def log(*args, **kwargs):
    # pylint: disable=missing-function-docstring
    # pylint: disable=protected-access
    for pbar in _PROGRESSBARS:
        pbar._tqdm_instance.clear()
    msg_frame = inspect.currentframe().f_back
    with temp_attr(
        LOGGER,
        "findCaller",
        lambda self, stack_info=None: (
            msg_frame.f_code.co_filename,
            msg_frame.f_lineno,
            msg_frame.f_code.co_name,
            None,
        ),
    ):
        LOGGER.log(*args, **kwargs)
    for pbar in _PROGRESSBARS:
        pbar._tqdm_instance.refresh()


def set_level(level: int):
    r"""Set logging level"""
    LOGGER.setLevel(level)


class Progressbar:
    r"""
    Context manager for creating progress bars compatible with the logging
    environment
    """

    def __init__(self, iterable, /, *, position=-1, **kwargs):
        self._iterable = iterable
        self._position = position
        self._kwargs = kwargs
        self._tqdm_instance = None

    def __enter__(self):
        # pylint: disable=no-member,attribute-defined-outside-init
        # ^ disable false positive linting errors
        self._tqdm_instance = tqdm(self._iterable, **self._kwargs)
        _PROGRESSBARS.insert(self._position % (len(_PROGRESSBARS) + 1), self)
        _refresh_progressbars()
        return self._tqdm_instance

    def __exit__(self, err_type, err, tb):
        _PROGRESSBARS.remove(self)
        self._tqdm_instance.close()
        _refresh_progressbars()


def temp_attr(obj: object, attr: str, value: Any):
    r"""
    Creates a context manager for setting transient object attributes.

    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(x=1)
    >>> with temp_attr(obj, 'x', 2):
    ...     print(obj.x)
    2
    >>> print(obj.x)
    1
    """

    class _TempAttr:
        def __init__(self):
            self.__original_value = None

        def __enter__(self):
            self.__original_value = getattr(obj, attr)
            setattr(obj, attr, value)

        def __exit__(self, *_):
            if getattr(obj, attr) == value:
                setattr(obj, attr, self.__original_value)
            else:
                warnings.warn(
                    f'Attribute "{attr}" changed while in context.'
                    " The new value will be kept.",
                )

    return _TempAttr()
