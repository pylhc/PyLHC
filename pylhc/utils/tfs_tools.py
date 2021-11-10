"""
Additional IO-Tools
---------------------

Additional tools for reading and writing TfsDataFrames,
that are not necessarily related to TFS-files.
"""
import pandas as pd
from pathlib import Path

from typing import Union

import h5py
from tfs import TfsDataFrame

import logging

LOGGER = logging.getLogger(__name__)


def write_hdf(path: Union[Path, str], df: TfsDataFrame, **kwargs):
    """Write TfsDataFrame to hdf5 file. The dataframe will be written into
    the group ``data``, the headers into the group ``headers``.
    Only one frame per file is allowed.

    Args:
        path (Path, str): Path of the output file.
        df (TfsDataFrame): TfsDataFrame to write.
        kwargs: kwargs to be passed to pandas ``DataFrame.to_hdf()``.
                ``key`` is not allowed and ``mode`` needs to be ``w`` if the
                output file already exists.
    """
    # Check for `key` kwarg (forbidden) ---
    if "key" in kwargs:
        raise AttributeError("The argument 'key' is not allowed here, "
                             "as only one TfsDataFrame per file is supported.")

    # Check for `mode` kwarg (allowed under circumstances but generally ignored) ---
    user_mode = kwargs.pop('mode', None)
    if user_mode is not None and user_mode != "w":
        if path.exists():
            raise AttributeError(f"'mode=\"{user_mode}\"' is not allowed here. "
                                 f"The output file at {str(path)} will always be overwritten!")
        LOGGER.warning(f"'mode=\"{user_mode}\"' is not allowed here. "
                       f"Mode \"w\" will be used.")

    # Actual writing of the output file ---
    df.to_hdf(path, key='data', mode='w', **kwargs)
    with h5py.File(path, mode="a") as hf:
        hf.create_group("headers")  # empty group in case of empty headers
        for key, value in df.headers.items():
            hf.create_dataset(f"headers/{key}", data=value)


def read_hdf(path: Union[Path, str]) -> TfsDataFrame:
    """Read TfsDataFrame from hdf5 file. The DataFrame needs to be stored
    in a group named ``data``, while the headers are stored in ``headers``.

    Args:
        path (Path, str): Path of the file to read.
    """
    df = pd.read_hdf(path, key="data")
    with h5py.File(path, mode="r") as hf:
        headers = hf.get('headers')
        headers = {k: headers[k][()] for k in headers.keys()}

    for key, value in headers.items():
        try:
            headers[key] = value.decode('utf-8')  # converts byte-strings back
        except AttributeError:
            pass  # probably numeric
    return TfsDataFrame(df, headers=headers)
