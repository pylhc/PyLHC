"""
Post Process DA
----------------------------------

Tools to process data after sixdb has calculated the
da. Includes functions for extracting data from database
as well as plotting of DA polar plots.
"""
import sqlite3 as sql
from pathlib import Path
from typing import Any, Tuple, Iterable, Union

import numpy as np
import pandas as pd
from generic_parser import DotDict
from matplotlib import pyplot as plt
from matplotlib import rcParams, lines as mlines
from omc3.plotting.utils import style as pstyle
from omc3.utils import logging_tools
from pylhc.constants.autosix import (
    get_database_path,
    get_tfs_da_path,
    get_tfs_da_seed_stats_path,
    get_tfs_da_angle_stats_path,
    get_autosix_results_path,
    HEADER_NTOTAL,
    HEADER_INFO,
    HEADER_HINT,
    MEAN,
    STD,
    MIN,
    MAX,
    N,
    SEED,
    ANGLE,
    ALOST1,
    ALOST2,
    AMP,
)
from scipy.interpolate import interp1d
from tfs import TfsDataFrame, write_tfs

from pylhc.sixdesk_tools.utils import StageSkip

LOG = logging_tools.get_logger(__name__)

DA_COLUMNS = (ALOST1, ALOST2)
INFO = (
    "Statistics over the N={n:d} {over:s} per {per:s}. "
    "The N-Columns indicate how many non-zero DA values were used."
)
HINT = "{param:s} {val:} is the respective value calculated over all other {param:s}s."
OVER_WHICH = {SEED: "angles", ANGLE: "seeds"}

# Fixed plot styles ---
COLOR_MEAN = "red"
COLOR_SEED = "grey"
COLOR_LIM = "black"
COLOR_FILL = "blue"
ALPHA_SEED = 0.5
ALPHA_FILL = 0.2
ALPHA_FILL_STD = 0.2


def post_process_da(jobname: str, basedir: Path):
    """ Post process the DA results into dataframes and DA plots. """
    LOG.info("Post-Processing Sixdesk Results.")
    try:
        df_da, df_angle, df_seed = create_da_tfs(jobname, basedir)
        create_polar_plots(jobname, basedir, df_da, df_angle)
    except Exception as e:
        raise StageSkip(f"Post Processing failed: {e.__traceback__}") from e
    LOG.info("Post-Processing finished.")


# Data Analysis ----------------------------------------------------------------


def create_da_tfs(jobname: str, basedir: Path) -> Tuple[TfsDataFrame, TfsDataFrame, TfsDataFrame]:
    """ Extracts data from db into dataframes, and writes and returns them."""
    LOG.info("Gathering DA data into tfs-files.")
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
    df_da = df_da.rename(
        columns={
            "seed": SEED,
            "angle": ANGLE,
            "alost1": ALOST1,
            "alost2": ALOST2,
            "Amin": f"{MIN}{AMP}",
            "Amax": f"{MAX}{AMP}",
        }
    )
    return TfsDataFrame(df_da)


def _create_stats_df(df: TfsDataFrame, parameter: str, global_index: Any = None):
    """ Calculates the stats over a given parameter """
    operation_map = DotDict({MEAN: np.mean, STD: np.std, MIN: np.min, MAX: np.max})

    pre_index = [] if global_index is None else [global_index]
    index = sorted(set(df[parameter]))
    n_total = sum(df[parameter] == index[0])

    df_stats = TfsDataFrame(
        index=pre_index + index,
        columns=[f"{fun}{al}" for al in DA_COLUMNS for fun in list(operation_map.keys()) + [N]],
    )
    df_stats.headers[HEADER_INFO] = INFO.format(
        over=OVER_WHICH[parameter], per=parameter.lower(), n=n_total
    )
    df_stats.headers[HEADER_NTOTAL] = n_total

    for col_da in DA_COLUMNS:
        for idx in index:
            mask = (df[parameter] == idx) & (df[col_da] != 0)
            df_stats.loc[idx, f"{N}{col_da}"] = sum(mask)
            for name, operation in operation_map.items():
                df_stats.loc[idx, f"{name}{col_da}"] = operation(df.loc[mask, col_da])
            for name, operation in operation_map.get_subdict([MIN, MAX]).items():
                df_stats.loc[idx, f"{name}{AMP}"] = operation(df.loc[mask, f"{name}{AMP}"])

        if global_index is not None:
            df_stats.loc[global_index, f"{N}{col_da}"] = sum(df_stats.loc[index, f"{N}{col_da}"])
            for name, operation in operation_map.items():
                df_stats.loc[global_index, f"{name}{col_da}"] = operation(
                    df_stats.loc[index, f"{name}{col_da}"]
                )
            for name, operation in operation_map.get_subdict([MIN, MAX]).items():
                df_stats.loc[global_index, f"{name}{AMP}"] = operation(
                    df_stats.loc[index, f"{name}{AMP}"]
                )
            df_stats.headers[HEADER_HINT] = HINT.format(param=parameter, val=global_index)

    return df_stats


# Single Plots -----------------------------------------------------------------


def create_polar_plots(jobname: str, basedir: Path, df_da: TfsDataFrame, df_angles: TfsDataFrame):
    """ Plotting loop over da-methods and wrapper so save plots. """
    LOG.info("Creating Polar Plots.")
    outdir_path = get_autosix_results_path(jobname, basedir)
    for da_col in DA_COLUMNS:
        fig = plot_polar(df_angles, da_col, jobname, df_da)
        fig.tight_layout(), fig.tight_layout()
        fig.savefig(outdir_path / fig.canvas.get_default_filename())
    # plt.show()


def plot_polar(
    df_angles: TfsDataFrame,
    da_col: str = ALOST2,
    jobname: str = "",
    df_da: TfsDataFrame = None,
    **kwargs,
) -> plt.Figure:
    """Create Polar Plot for DA analysis data.

    Keyword arguments are all optional.

    Args:
        df_angles (TfsDataFrame): Dataframe with the statistics (min, max, mean) per angle
        da_col (str): DA-Column name from sixdesk analysis to be used , e.g. ``ALOST2``.
                      (optional, default: ``ALOST2``)
        jobname (str): Name of the job. Used in window title only (optional).
        df_da (TfsDataFrame): Full DA analysis result. If given, plots
                              the individual DA results per seed. (optional)

    Keyword Arguments:
        plot_styles (Iterable[str]): Iterable over plots styles to be applied, default: 'standard'
        interpolated (bool): If true, uses interpolation to plot the lines curved
        fill (bool): If true, fills the area between min and max with light blue
        angle_ticks (Iterable[numeric]): Positions in degree of the angle ticks (and lines)
        amplitude ticks (Iterable[numeric]): Positions of the amplitude ticks.
        remaining args: any rcParams to be set.

    Returns:
        Figure of the polar plot.
    """
    interpolated: bool = kwargs.pop("interpolated", True)
    fill: bool = kwargs.pop("fill", df_da is None)
    angle_ticks: Iterable[np.numeric] = kwargs.pop("angle_ticks", None)
    amplitude_ticks: Iterable[np.numeric] = kwargs.pop("amplitude_ticks", None)
    plot_styles: Iterable[Union[Path, str]] = kwargs.pop("plot_styles", "standard")

    if "lines.marker" not in kwargs:
        kwargs["lines.marker"] = "None"
    pstyle.set_style(plot_styles, kwargs)
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "polar"})
    fig.canvas.set_window_title(f"{jobname} polar plot for {da_col}")

    angles = np.deg2rad(df_angles.index)
    da_min, da_mean, da_max, da_std = (
        df_angles[f"{name}{da_col}"] for name in (MIN, MEAN, MAX, STD)
    )

    seed_h, seed_l = _plot_seeds(ax, df_da, da_col, interpolated)
    if interpolated:
        mean_h, max_h = _plot_interpolated(ax, angles, da_min, da_mean, da_max, da_std, fill)
    else:
        mean_h, max_h = _plot_straight(ax, angles, da_min, da_mean, da_max, da_std, fill)

    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlim([0, None])
    ax.set_xlabel(r"$\sigma_{x}~[\sigma_{nominal}]$", labelpad=15)
    ax.set_ylabel(r"$\sigma_{y}~[\sigma_{nominal}]$", labelpad=20)

    if angle_ticks is not None:
        ax.set_xticks(np.deg2rad(angle_ticks))

    if amplitude_ticks is not None:
        ax.set_yticks(amplitude_ticks)
    ax.tick_params(labelright=True, labelleft=True)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.9, 0.95),
        bbox_transform=fig.transFigure,
        # frameon=False,
        handles=seed_h + [mean_h, max_h],
        labels=seed_l + ["Mean DA", "Limits"],
        ncol=1,
    )

    return fig


def _plot_seeds(ax, df_da, da_col, interpolated):
    """Add the Seed lines to the polar plots, if df_da is given."""
    if df_da is not None:
        seed_h = None
        for seed in sorted(set(df_da[SEED])):
            seed_mask = df_da[SEED] == seed
            angles = np.deg2rad(df_da.loc[seed_mask, ANGLE])
            da_data = df_da.loc[seed_mask, da_col]
            da_data.loc[da_data == 0] = np.NaN
            if interpolated:
                seed_h, _, _ = _interpolated_line(
                    ax,
                    angles,
                    da_data,
                    c=COLOR_SEED,
                    ls="-",
                    label=f"Seed {seed:d}",
                    alpha=ALPHA_SEED,
                )
            else:
                (seed_h,) = ax.plot(
                    angles, da_data, c=COLOR_SEED, ls="-", label=f"Seed {seed:d}", alpha=ALPHA_SEED
                )
        return [seed_h], ["DA per Seed"]
    return [], []


def _plot_interpolated(ax, angles, da_min, da_mean, da_max, da_std, fill):
    """Plot interpolated DA lines and areas."""
    _, _, ip_min = _interpolated_line(ax, angles, da_min, c=COLOR_LIM, ls="--", label="Minimum DA")
    max_h, ip_x, ip_max = _interpolated_line(
        ax, angles, da_max, c=COLOR_LIM, ls="--", label="Maximum DA"
    )
    if fill:
        ax.fill_between(ip_x, ip_min, ip_max, color=COLOR_FILL, alpha=ALPHA_FILL)
        _, ip_std_min = _interpolated_coords(angles, da_mean - da_std)
        _, ip_std_max = _interpolated_coords(angles, da_mean + da_std)
        ax.fill_between(ip_x, ip_std_min, ip_std_max, color=COLOR_FILL, alpha=ALPHA_FILL_STD)

    mean_h, _, _ = _interpolated_line(ax, angles, da_mean, c=COLOR_MEAN, ls="-", label="Mean DA")
    return mean_h, max_h


def _plot_straight(ax, angles, da_min, da_mean, da_max, da_std, fill):
    """Plot straight DA lines and areas."""
    (_,) = ax.plot(angles, da_min, c=COLOR_LIM, ls="--", label="Minimum DA")
    (max_h,) = ax.plot(angles, da_max, c=COLOR_LIM, ls="--", label="Maximum DA")
    if fill:
        ax.fill_between(
            angles,
            da_min.astype(float),
            da_max.astype(float),  # weird conversion to obj otherwise
            color=COLOR_FILL,
            alpha=ALPHA_FILL,
        )
        ax.fill_between(
            angles,
            (da_mean - da_std).astype(float),
            (da_mean + da_std).astype(float),
            color=COLOR_FILL,
            alpha=ALPHA_FILL_STD,
        )
    (mean_h,) = ax.plot(angles, da_mean, c=COLOR_MEAN, ls="-", label="Mean DA")
    return mean_h, max_h


def _interpolated_line(ax, x, y, npoints: int = 100, **kwargs):
    """Plot a line that interpolates linearly between points.
    Useful for polar plots with sparse points."""
    ls = kwargs.pop("linestyle", kwargs.pop("ls", rcParams["lines.linestyle"]))
    marker = kwargs.pop("marker", rcParams["lines.marker"])
    label = kwargs.pop("label")

    ip_x, ip_y = _interpolated_coords(x, y, npoints)
    (line_h,) = ax.plot(ip_x, ip_y, marker="None", ls=ls, label=f"_{label}_line", **kwargs)

    if marker.lower() not in ["none", ""]:
        ax.plot(x, y, ls="None", marker=marker, label=f"_{label}_markers", **kwargs)

    # fake handle for legend
    handle = mlines.Line2D([], [], color=line_h.get_color(), ls=ls, marker=marker, label=label)
    return handle, ip_x, ip_y


def _interpolated_coords(x, y, npoints: int = 100):
    """ Do linear interpolation between points. """
    ip_x = np.linspace(min(x), max(x), npoints)
    ip_y = interp1d(x, y)(ip_x)
    return ip_x, ip_y
