"""
Diff Files
-------------------------------

Functions to get the difference between two tfs files or dataframes.
"""
import tfs
from generic_parser import entrypoint, EntryPointParameters
import pandas as pd
from utils.diff_tools import df_diff, df_ang_diff, df_ratio, df_rel_diff, df_error_diff


KIND_MAP = {
    'absolute': df_diff,
    'relative': df_rel_diff,
    'error': df_error_diff,
    'angle': df_ang_diff,
    'ratio': df_ratio,
}


def get_params():
    params = EntryPointParameters()
    params.add_parameter(flags=["--df1"], name="df1",
                         required=True,
                         help="First dataframe, Minuend or Dividend")
    params.add_parameter(flags=["--df2"], name="df2",
                         required=True,
                         help="Second dataframe, Subtrahend or Divisor")
    params.add_parameter(flags=["--columns"], name="columns",
                         required=True, nargs="+", type=str,
                         help="List of columns to get the difference of")
    params.add_parameter(flags=["--kind"], name="kind",
                         nargs="+", type=str,
                         choices=['absolute', 'relative', 'error', 'angle', 'ratio'],
                         help="Kind of difference. Can be given per column. Defaults to 'absoulte'.")
    params.add_parameter(flags=["--prefix"], name="prefix",
                         type=str, default="",
                         help="Prefix for difference columns.")
    params.add_parameter(flags=["--index"], name="index",
                         type=str,
                         help="Index column - most likely needed when reading/writing files")
    params.add_parameter(flags=["--keep"], name="keep_columns",
                         nargs="+", type=str, default=[],
                         help="Additional columns to keep in the returned dataframe")
    params.add_parameter(flags=["--out"], name="out_file",
                         type=str,
                         help="If given, writes the result into this file")
    return params


@entrypoint(get_params(), strict=True)
def get_diff_two_dataframes(opt):
    """ Get the difference of common elements of specific columns between two dataframes.
    Merges on index.

    Keyword Args:
        *--Required--*
        - **columns** *(str)*: List of columns to get the difference of

          Flags: **['--columns']**
        - **df1**: First dataframe, Minuend or Dividend

          Flags: **['--df1']**
        - **df2**: Second dataframe, Subtrahend or Divisor

          Flags: **['--df2']**

        *--Optional--*
        - **index** *(str)*: Index column - most likely needed when reading/writing files

          Flags: **['--index']**
        - **keep_columns** *(str)*: Additional columns to keep in the returned dataframe

          Flags: **['--keep']**
          Default: ``[]``
        - **kind** *(str)*: Kind of difference. Can be given per column. Defaults to 'absoulte'.

          Flags: **['--kind']**
          Choices: ``['absolute', 'relative', 'error', 'angle', 'ratio']``
        - **out_file** *(str)*: If given, writes the result into this file

          Flags: **['--out']**
        - **prefix** *(str)*: Prefix for difference columns.

          Flags: **['--prefix']**
          Default: ````

        Returns:
            DataFrame containing difference columns and kept columns.
    """
    # convert from files to dataframes
    df1 = _get_dataframe(opt.df1, opt.index)
    df2 = _get_dataframe(opt.df2, opt.index)

    # check input
    _check_for_missing_columns(df1, df2, opt.columns)
    _check_for_missing_columns(df1, df2, opt.keep_columns)

    # merge dataframes
    merged = pd.merge(df1, df2, how='inner',
                      left_index=True, right_index=True,
                      suffixes=('_df1', '_df2'))

    # calculate difference
    if opt.kind is None:
        opt.kind = ['absolute'] * len(opt.columns)
    elif len(opt.kind) == 1 and len(opt.columns > 1):
        opt.kind = [opt.kind] * len(opt.columns)
    elif len(opt.kind) != len(opt.columns):
        raise ValueError(
            "The length of the differece kinds array needs to correspond to the number of columns."
        )

    for idx, col in enumerate(opt.columns):
        merged[f"{opt.prefix}{col}"] = KIND_MAP[opt.kind[idx]](merged, f'{col}_df1', f'{col}_df2')

    # copy columns to be kept
    for col in opt.keep_columns:
        for suffix in ["", "_df1", "_df2"]:
            try:
                merged[col] = merged[f"{col}{suffix}"]
            except KeyError:
                pass
            else:
                break

    # prepare output
    merged = merged.loc[:, opt.keep_columns + [opt.prefix + c for c in opt.columns]]
    if opt.out_file:
        tfs.write(opt.out_file, merged, save_index=opt.index)
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
            "The following columns can not be found in one or both of the dataframes: {:}".format(
                list(set(missing_columns)))
        )


# Script #######################################################################


if __name__ == '__main__':
    get_diff_two_dataframes()
