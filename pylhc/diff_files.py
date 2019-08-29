"""
Module tfs_files.diff_files
-------------------------------

Functions to get the difference between two tfs files or dataframes.
This is very general, i.e. not as results oriented as ``getdiff.py``.
"""
import tfs
import pandas as pd
import numpy as np
from utils.diff_tools import df_diff, df_ang_diff, df_ratio, df_rel_diff, df_error_diff

KIND_MAP ={
    'absolute': df_diff,
    'relative': df_rel_diff,
    'error': df_error_diff,
    'circular': df_ang_diff,
    'ratio': df_ratio,
}


def get_diff_two_dataframes(df1, df2, columns, kind=None, prefix="", index=None, keep_columns=(), out_file=None):
    """ Get the difference of common elements of specific columns between two dataframes.

    Merges on index.

        Args:
            df1 (DataFrame or Path): First dataframe, Minuend or Dividend
            df2 (DataFrame or Path): Second dataframe, Subtrahend or Divisor
            columns (list of stings): List of columns to get the difference of
            kind (list of strings): defines the kind of difference 'absolute', 'relative', 'error', 'circular', 'ratio'
            prefix (str): Prefix for difference columns (default: "")
            index (str): index column - most likely needed when reading/writing files
            keep_columns (list of strings): additional columns to keep in the returned dataframe
            out_file (Path): if given, writes the result into this file

        Returns:
            DataFrame containing difference columns and kept columns.
    """
    # convert from files to dataframes
    df1 = _get_dataframe(df1, index)
    df2 = _get_dataframe(df2, index)

    # check input
    _check_for_missing_columns(df1, df2, columns)
    _check_for_missing_columns(df1, df2, keep_columns)

    # merge dataframes
    merged = pd.merge(df1, df2, how='inner',
                      left_index=True, right_index=True,
                      suffixes=('_df1', '_df2'))

    # calculate difference
    if kind is None:
        kind = ['absolute'] * len(columns)
    else:
        if len(kind) != len(columns):
            raise ValueError(
                "The length of the differece kinds array needs to correspond to the number of columns."
            )

    for idx, col in enumerate(columns):
        merged[prefix + col] = KIND_MAP[kind[idx]](merged, f'{col}_df1', f'{col}_df2')

    # copy columns to be kept
    for col in keep_columns:
        for suffix in ["", "_df1", "_df2"]:
            try:
                merged[col] = merged[col + suffix]
            except KeyError:
                pass
            else:
                break

    # prepare output
    merged = merged.loc[:, keep_columns + [prefix + c for c in columns]]
    if out_file:
        tfs.write(out_file, merged, save_index=index)
    return merged


# Helpers #####################################################################


def _get_dataframe(tfs_df, index):
    try:
        return tfs.read(tfs_df, index=index)
    except TypeError:
        return tfs_df


def _check_for_missing_columns(df1, df2, columns):
    missing_columns = [col for col in columns for df in [df1, df2] if col not in df.columns]
    if any(missing_columns):
        raise KeyError(
            "The following columns can not be found in either dataframe: {:}".format(
                list(set(missing_columns)))
        )
