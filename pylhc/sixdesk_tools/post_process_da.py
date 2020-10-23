import sqlite3 as sql
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from generic_parser import DotDict
from matplotlib import pyplot as plt
from matplotlib import rcParams, lines as mlines
from scipy.interpolate import interp1d
from tfs import TfsDataFrame, write_tfs

from pylhc.constants.autosix import (
    get_database_path,
    get_tfs_da_path, get_tfs_da_seed_stats_path, get_tfs_da_angle_stats_path,
    get_autosix_results_path
)

HEADER_NTOTAL, HEADER_INFO, HEADER_HINT = "NTOTAL", "INFO", "HINT"
MEAN, STD, MIN, MAX, N = 'MEAN', 'STD', 'MIN', 'MAX', "N"
SEED, ANGLE, ALOST1, ALOST2, AMP = 'SEED', 'ANGLE', 'ALOST1', 'ALOST2', 'A'
DA_COLUMNS = (ALOST1, ALOST2)

INFO = ('Statistics over the N={n:d} {over:s} per {per:s}. '
        'The N-Columns indicate how many non-zero DA values were used.')

HINT = '{param:s} {val:} is the respective value calculated over all other {param:s}s.'

OVER_WHICH = {SEED: 'angles', ANGLE: 'seeds'}

COLOR_MEAN = 'red'
COLOR_SEED = 'grey'
COLOR_LIM = 'black'
ALPHA_SEED = 0.5


def post_process_da(jobname: str, basedir: Path):
    df_da, df_angle, df_seed = create_da_tfs(jobname, basedir)
    create_polar_plots(jobname, basedir, df_da, df_angle)


# Data Analysis ----------------------------------------------------------------

def create_da_tfs(jobname: str, basedir: Path) -> Tuple[TfsDataFrame, TfsDataFrame, TfsDataFrame]:
    df_da = extract_da_data_from_db(jobname, basedir)

    df_angle = _create_stats_df(df_da, ANGLE)
    df_seed = _create_stats_df(df_da, SEED, global_index=0)

    write_tfs(get_tfs_da_path(jobname, basedir), df_da)
    write_tfs(get_tfs_da_angle_stats_path(jobname, basedir), df_angle, save_index=ANGLE)
    write_tfs(get_tfs_da_seed_stats_path(jobname, basedir), df_seed, save_index=SEED)
    
    return df_da, df_angle, df_seed


def extract_da_data_from_db(jobname: str, basedir: Path) -> TfsDataFrame:
    """ Extract data directly from the database. """
    db_path = get_database_path(jobname, basedir)
    db = sql.connect(db_path)
    df_da = pd.read_sql(
        "SELECT seed, angle, alost1, alost2, Amin, Amax FROM da_post ORDER BY seed, angle", db
    )
    df_da = df_da.rename(columns={'seed': SEED, 'angle': ANGLE,
                                  'alost1': ALOST1, 'alost2': ALOST2,
                                  'Amin': f'{MIN}{AMP}', 'Amax': f'{MAX}{AMP}'})
    return TfsDataFrame(df_da)


def _create_stats_df(df: TfsDataFrame, parameter: str, global_index: Any = None):
    """ Calculates the stats over a given parameter """
    operation_map = DotDict({MEAN: np.mean, STD: np.std, MIN: np.min, MAX: np.max})

    pre_index = [] if global_index is None else [global_index]
    index = sorted(set(df[parameter]))
    n_total = sum(df[parameter] == index[0])

    df_stats = TfsDataFrame(
        index=pre_index + index,
        columns=[f'{fun}{al}' for al in DA_COLUMNS for fun in list(operation_map.keys()) + [N]]
    )
    df_stats.headers[HEADER_INFO] = INFO.format(over=OVER_WHICH[parameter],
                                                per=parameter.lower(),
                                                n=n_total)
    df_stats.headers[HEADER_NTOTAL] = n_total

    for col_da in DA_COLUMNS:
        for idx in index:
            mask = (df[parameter] == idx) & (df[col_da] != 0)
            df_stats.loc[idx, f'{N}{col_da}'] = sum(mask)
            for name, operation in operation_map.items():
                df_stats.loc[idx, f'{name}{col_da}'] = operation(df.loc[mask, col_da])
            for name, operation in operation_map.get_subdict([MIN, MAX]).items():
                df_stats.loc[idx, f'{name}{AMP}'] = operation(df.loc[mask, f'{name}{AMP}'])

        if global_index is not None:
            df_stats.loc[global_index, f'{N}{col_da}'] = sum(df_stats.loc[index, f'{N}{col_da}'])
            for name, operation in operation_map.items():
                df_stats.loc[global_index, f'{name}{col_da}'] = operation(df_stats.loc[index, f'{name}{col_da}'])
            for name, operation in operation_map.get_subdict([MIN, MAX]).items():
                df_stats.loc[global_index, f'{name}{AMP}'] = operation(df_stats.loc[index, f'{name}{AMP}'])
            df_stats.headers[HEADER_HINT] = HINT.format(param=parameter, val=global_index)

    return df_stats


# Single Plots -----------------------------------------------------------------

def create_polar_plots(jobname: str, basedir: Path, df_da: TfsDataFrame, df_angles: TfsDataFrame):
    outdir_path = get_autosix_results_path(jobname, basedir)
    for da_col in DA_COLUMNS:
        fig = plot_polar(df_angles, da_col, jobname, df_da)
        fig.tight_layout(), fig.tight_layout()
        fig.savefig(outdir_path / fig.canvas.get_default_filename())
    plt.show()


def plot_polar(df_angles: TfsDataFrame, da_col: str, jobname: str,
               df_da: TfsDataFrame = None, interpolated: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': 'polar'})
    fig.canvas.set_window_title(f"{jobname} polar plot for {da_col}")

    seed_h = []
    seed_l = []
    if df_da is not None:
        for seed in sorted(set(df_da[SEED])):
            seed_mask = df_da[SEED] == seed
            angles = np.deg2rad(df_da.loc[seed_mask, ANGLE])
            da_data = df_da.loc[seed_mask, da_col]
            da_data.loc[da_data == 0] = np.NaN
            if interpolated:
                seed_h = _interpolated_line(ax, angles, da_data, c=COLOR_SEED, ls='-', label=f'Seed {seed:d}', alpha=ALPHA_SEED)
            else:
                seed_h, = ax.plot(angles, da_data, c=COLOR_SEED, ls='-', label=f'Seed {seed:d}', alpha=ALPHA_SEED)
        seed_h = [seed_h]
        seed_l = ['DA per Seed']

    angles = np.deg2rad(df_angles.index)
    da_min, da_mean, da_max = (df_angles[f'{name}{da_col}'] for name in (MIN, MEAN, MAX))
    if interpolated:
        min_h = _interpolated_line(ax, angles, da_min, c=COLOR_LIM, ls='--', label='Minimum DA')
        max_h = _interpolated_line(ax, angles, da_max, c=COLOR_LIM, ls='--', label='Maximum DA')
        mean_h = _interpolated_line(ax, angles, da_mean, c=COLOR_MEAN, ls='-', label='Mean DA')
    else:
        min_h, = ax.plot(angles, da_min, c=COLOR_LIM, ls='--', label='Minimum DA')
        max_h, = ax.plot(angles, da_max, c=COLOR_LIM, ls='--', label='Maximum DA')
        # ax.fill_between(angles, da_min.astype(float), da_max.astype(float),  # weird conversion to obj otherwise
        #                 color='blue', alpha=0.2)
        mean_h, = ax.plot(angles, da_mean, c=COLOR_MEAN, ls='-', label='Mean DA')

    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlim([0, None])
    ax.set_xlabel(r'$\sigma_{x}~[\sigma_{nominal}]$', labelpad=15)
    ax.set_ylabel(r'$\sigma_{y}~[\sigma_{nominal}]$')

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(.9, .95),
        bbox_transform=fig.transFigure,
        # frameon=False,
        handles=seed_h + [mean_h, max_h],
        labels=seed_l + ['Mean DA', 'Limits'],
        ncol=1,
    )

    return fig


def _interpolated_line(ax, x, y, npoints: int = 100, **kwargs):
    """ Plot a line that interpolates linearly between points.
    Useful for polar plots with sparse points. """
    ls = kwargs.pop('linestyle', kwargs.pop('ls', rcParams['lines.linestyle']))
    marker = kwargs.pop('marker', rcParams['lines.marker'])
    label = kwargs.pop('label')

    handle_line = None
    if ls.lower() not in ['none', '']:
        ip_x = np.linspace(min(x), max(x), npoints)
        ip_y = interp1d(x, y)(ip_x)
        handle_line, = ax.plot(ip_x, ip_y, marker='None', ls=ls, label=f'_{label}_line', **kwargs)

    handle_marker = None
    if marker.lower() not in ['none', '']:
        handle_marker, = ax.plot(x, y, ls='None', marker=marker, label=f'_{label}_markers', **kwargs)

    color = None
    if handle_line:
        color = handle_line.get_color()
    elif handle_marker:
        color = handle_marker.get_color()

    return mlines.Line2D([], [], color=color, ls=ls, marker=marker, label=label)