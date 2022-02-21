"""
BSRT Analysis
-------------

Top-level script to query BRST data, then perform an analysis and output the generated plots. The
script runs through the following steps:
- Processes the output files of ``BSRT_logger.py`` for a given timeframe, returns them in as a
`TfsDataFrame` for further processing.
- Additionally, plots for quick checks of fit parameters, auxiliary variables and beam evolution
are generated.
- If provided a `TfsDataFrame` file with timestamps, plots of the 2D distribution and comparison
of fit parameters to cross sections are added.
"""
import datetime
import glob
import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import pytz
import tfs
from generic_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools, time_tools

from pylhc.constants.general import TFS_SUFFIX, TIME_COLUMN

LOG = logging_tools.get_logger(__name__)
PLOT_FILE_SUFFIX = ".pdf"
BSRT_FESA_TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
OLD_FILENAMING_CONV = "{}-{}-{}-{}-{}-{}.{}"
NEW_FILENAMING_CONV = "{}_{}_{}@{}_{}_{}_{}"


def get_params():
    return EntryPointParameters(
        directory=dict(
            flags=["-d", "--directory"],
            required=True,
            type=str,
            help="Directory containing the logged BSRT files.",
        ),
        beam=dict(
            flags=["-b", "--beam"],
            required=True,
            choices=["B1", "B2"],
            type=str,
            help="Beam for which analysis is performed.",
        ),
        outputdir=dict(
            flags=["-o", "--outputdir"],
            type=str,
            default=None,
            help=(
                "Directory in which plots and dataframe will be saved in. If omitted, "
                "no data will be saved."
            ),
        ),
        starttime=dict(
            flags=["--starttime"],
            type=int,
            help="Start of time window for analysis in milliseconds UTC.",
        ),
        endtime=dict(
            flags=["--endtime"],
            type=int,
            help="End of time window for analysis in milliseconds UTC.",
        ),
        kick_df=dict(
            flags=["--kick_df"],
            default=None,
            help=(
                f"TFS with column {TIME_COLUMN} with time stamps to be added in the plots. "
                f"Additionally, cross section at these timestamps will be plotted.",
            ),
        ),
        show_plots=dict(flags=["--show_plots"], type=bool, default=False, help="Show BSRT plots."),
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting BSRT analysis.")
    LOG.info("Indexing files in directory.")
    files_df = _load_files_in_df(opt)
    LOG.info("Select files based on provided timestamps.")
    files_df = _select_files(opt, files_df)
    LOG.info("Load BSRT files in selected time frame.")
    bsrt_df = _load_pickled_data(opt, files_df)
    results = {"bsrt_df": bsrt_df}

    if not opt["show_plots"] and opt["outputdir"] is None:
        LOG.info("Neither plot display nor outputdir was selected. Plotting is omitted")
        return results

    if opt["kick_df"] is not None and isinstance(opt["kick_df"], str):
        opt["kick_df"] = tfs.read(opt["kick_df"], index=TIME_COLUMN)

    LOG.info("Plotting Fitvariables.")
    results.update(fitvariables=plot_fit_variables(opt, bsrt_df))
    LOG.info("Plotting full cross section.")
    results.update(full_crosssection=plot_full_crosssection(opt, bsrt_df))
    LOG.info("Plotting auxiliary variables.")
    results.update(auxiliary_variables=plot_auxiliary_variables(opt, bsrt_df))
    if opt["kick_df"] is not None:
        LOG.info("Plotting cross section for timesteps.")
        results.update(crosssection_for_timesteps=plot_crosssection_for_timesteps(opt, bsrt_df))
    return results


# File Name Functions ----------------------------------------------------------


def _get_bsrt_logger_fname(beam, timestamp) -> str:
    return f"data_BSRT_{beam}_{timestamp}.dat.gz"


def _get_bsrt_tfs_fname(beam) -> str:
    return f"data_BSRT_{beam}{TFS_SUFFIX}"


def _get_fitvar_plot_fname(beam) -> str:
    return f"plot_BSRT_FitVariables_{beam}{PLOT_FILE_SUFFIX}"


def _get_2dcrossection_plot_fname(beam) -> str:
    return f"plot_BSRT_2DCross_section_{beam}{PLOT_FILE_SUFFIX}"


def _get_crossection_plot_fname(beam, timestamp) -> str:
    return f"plot_BSRT_Cross_section_{timestamp}_{beam}{PLOT_FILE_SUFFIX}"


def _get_auxiliary_var_plot_fname(beam) -> str:
    return f"plot_BSRT_auxVariables_{beam}{PLOT_FILE_SUFFIX}"


# File Handling  ---------------------------------------------------------------


def _select_files(opt, files_df):
    if opt["endtime"] is not None and opt["starttime"] is not None:
        assert opt["endtime"] >= opt["starttime"]

    indices = []
    for time, fct in zip(
        [opt["starttime"], opt["endtime"]], ["first_valid_index", "last_valid_index"]
    ):
        indices.append(
            _get_closest_index(files_df, time if time is not None else getattr(files_df, fct)())
        )

    return files_df.iloc[indices[0] : indices[1] + 1]


def _load_files_in_df(opt):
    files_df = pd.DataFrame(
        data={"FILES": glob.glob(str(Path(opt.directory) / _get_bsrt_logger_fname(opt.beam, "*")))}
    )

    files_df = files_df.assign(
        TIMESTAMP=[
            _get_timestamp_from_name(
                Path(f).name,
                _get_bsrt_logger_fname(
                    opt.beam, NEW_FILENAMING_CONV if ("@" in Path(f).name) else OLD_FILENAMING_CONV
                ),
            )
            for f in files_df["FILES"]
        ]
    )
    files_df = files_df.assign(TIME=[f.timestamp() for f in files_df["TIMESTAMP"]])

    files_df = files_df.sort_values(by=["TIME"]).reset_index(drop=True).set_index("TIME")
    return files_df


def _get_closest_index(df, time):
    return df.index.get_loc(time, method="nearest")


def _get_timestamp_from_name(name, formatstring):
    year, month, day, hour, minute, second, microsecond = map(int, parse.parse(formatstring, name))
    return datetime.datetime(
        year, month, day, hour, minute, second, microsecond, tzinfo=pytz.timezone("UTC")
    )


def _check_and_fix_entries(entry):
    # pd.to_csv does not handle np.array as entries nicely, converting to list circumvents this
    for key, val in entry.items():
        if isinstance(val, (np.ndarray, tuple)):
            entry[key] = list(val)
        if np.array(val).size == 0:
            entry[key] = np.nan
    return entry


def _load_pickled_data(opt, files_df):
    merged_df = pd.DataFrame()
    for bsrtfile in files_df["FILES"]:
        data = pickle.load(gzip.open(bsrtfile, "rb"))
        for entry in data:
            entry = _check_and_fix_entries(entry)
            merged_df = merged_df.append(entry, ignore_index=True)

    merged_df = merged_df.set_index(
        pd.to_datetime(merged_df["acqTime"], format=BSRT_FESA_TIME_FORMAT)
    )
    merged_df.index.name = "TimeIndex"
    merged_df = merged_df.drop_duplicates(subset=["acqCounter", "acqTime"])
    if opt.outputdir is not None:
        merged_df.to_csv(Path(opt.outputdir, _get_bsrt_tfs_fname(opt.beam)))

    return merged_df


# Plotting Functions  ----------------------------------------------------------


def _add_kick_lines(ax, df):
    if df is not None:
        for idx, _ in df.iterrows():
            ax.axvline(x=time_tools.cern_utc_string_to_utc(idx), color="red", linestyle="--")


def _fit_var(ax, bsrt_df, plot_dict, opt):

    ax[plot_dict["idx"]].plot(
        bsrt_df.index, [entry[plot_dict["fitidx"]] for entry in bsrt_df["lastFitResults"]]
    )
    ax[plot_dict["idx"]].set_title(plot_dict["title"])
    ax[plot_dict["idx"]].set_ylim(bottom=plot_dict["bottom"])
    _add_kick_lines(ax[plot_dict["idx"]], opt["kick_df"])


def plot_fit_variables(opt, bsrt_df):

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 9), sharex=True, constrained_layout=True)

    plot_dicts = [
        {"idx": (0, 0), "fitidx": 2, "title": "Horizontal Amplitude", "bottom": 0},
        {"idx": (0, 1), "fitidx": 3, "title": "Horizontal Center", "bottom": None},
        {"idx": (0, 2), "fitidx": 4, "title": "Horizontal Sigma", "bottom": 0},
        {"idx": (1, 0), "fitidx": 7, "title": "Vertical Amplitude", "bottom": 0},
        {"idx": (1, 1), "fitidx": 8, "title": "Vertical Center", "bottom": None},
        {"idx": (1, 2), "fitidx": 9, "title": "Vertical Sigma", "bottom": 0},
    ]

    [_fit_var(ax, bsrt_df, plot_dict, opt) for plot_dict in plot_dicts]

    if opt["outputdir"] is not None:
        plt.savefig(Path(opt.outputdir, _get_fitvar_plot_fname(opt.beam)))
    if opt["show_plots"]:
        plt.show()
    return fig


def flattend_column(df, col):
    flat_column = []
    for _, entry in df.iterrows():
        flat_column = [*flat_column, *entry[col]]
    return flat_column


def pcolormesh_irregulargrid(ax, df, x_column, y_column, z_column):
    df["Starttime"] = df[x_column] - 0.5 * df[x_column].diff().fillna(pd.Timedelta(seconds=0))
    df["Endtime"] = df[x_column] + 0.5 * df[x_column].diff().shift(periods=-1).fillna(
        pd.Timedelta(seconds=0)
    )
    vmax = np.max(flattend_column(df, z_column))
    vmin = np.min(flattend_column(df, z_column))
    for _, row in df.iterrows():
        ax.pcolormesh(
            [row["Starttime"], row["Endtime"]],
            np.concatenate(
                [
                    row[y_column][0],
                    row[y_column][:-1] + 0.5 * np.diff(row[y_column]),
                    row[y_column][-1],
                ],
                axis=None,
            ),
            np.array([row[z_column]]).T,
            vmin=vmin,
            vmax=vmax,
            cmap="inferno",
        )


def _full_crossection(ax, bsrt_df, plot_dict, opt):
    pcolormesh_irregulargrid(
        ax,
        bsrt_df.reset_index(),
        "TimeIndex",
        f'projPositionSet{plot_dict["idx"]}',
        f'projDataSet{plot_dict["idx"]}',
    )
    ax.plot(
        bsrt_df.index,
        [entry[plot_dict["fitresult"]] for entry in bsrt_df["lastFitResults"]],
        color="white",
        linewidth=0.5,
    )
    ax.plot(
        bsrt_df.index,
        [
            entry[plot_dict["fitresult"]] + entry[plot_dict["fiterror"]]
            for entry in bsrt_df["lastFitResults"]
        ],
        color="white",
        linestyle="--",
        linewidth=0.3,
    )
    ax.plot(
        bsrt_df.index,
        [
            entry[plot_dict["fitresult"]] - entry[plot_dict["fiterror"]]
            for entry in bsrt_df["lastFitResults"]
        ],
        color="white",
        linestyle="--",
        linewidth=0.3,
    )
    ax.set_title(plot_dict["title"])
    _add_kick_lines(ax, opt["kick_df"])


def plot_full_crosssection(opt, bsrt_df):

    plot_dicts = [
        {"idx": 1, "fitresult": 3, "fiterror": 4, "title": "Horizontal Cross section"},
        {"idx": 2, "fitresult": 8, "fiterror": 9, "title": "Vertical Cross section"},
    ]

    fig, ax = plt.subplots(nrows=len(plot_dicts), ncols=1, figsize=(18, 9), constrained_layout=True)
    [_full_crossection(axis, bsrt_df, plot_dict, opt) for axis, plot_dict in zip(ax, plot_dicts)]

    if opt["outputdir"] is not None:
        plt.savefig(Path(opt.outputdir, _get_2dcrossection_plot_fname(opt.beam)))
    if opt["show_plots"]:
        plt.show()
    return fig


def _gauss(x, *p):
    a, b, c = p
    return a * np.exp(-((x - b) ** 2) / (2.0 * c ** 2.0))


def _reshaped_imageset(df):
    return np.reshape(
        df["imageSet"], (df["acquiredImageRectangle"][3], df["acquiredImageRectangle"][2])
    )


def plot_crosssection_for_timesteps(opt, bsrt_df):
    kick_df = opt["kick_df"]
    figlist = []
    for idx, _ in kick_df.iterrows():
        timestamp = pd.to_datetime(time_tools.cern_utc_string_to_utc(idx))

        data_row = bsrt_df.iloc[_get_closest_index(bsrt_df, timestamp)]
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 9), constrained_layout=True)

        fig.suptitle(f"Timestamp: {timestamp}")

        ax[0].imshow(_reshaped_imageset(data_row), cmap="hot", interpolation="nearest")
        ax[0].set_title(f"2D Pixel count")

        ax[1].plot(data_row["projPositionSet1"], data_row["projDataSet1"], color="darkred")
        ax[1].plot(
            data_row["projPositionSet1"],
            _gauss(
                np.array(data_row["projPositionSet1"]),
                data_row["lastFitResults"][2],
                data_row["lastFitResults"][3],
                data_row["lastFitResults"][4],
            ),
            color="darkgreen",
            label="Gaussian Fit",
        )
        ax[1].set_ylim(bottom=0)
        ax[1].legend()
        ax[1].set_title("Horizontal Projection")

        ax[2].plot(data_row["projPositionSet2"], data_row["projDataSet2"], color="darkred")
        ax[2].plot(
            data_row["projPositionSet2"],
            _gauss(
                np.array(data_row["projPositionSet2"]),
                data_row["lastFitResults"][7],
                data_row["lastFitResults"][8],
                data_row["lastFitResults"][9],
            ),
            color="darkgreen",
            label="Gaussian Fit",
        )
        ax[2].set_ylim(bottom=0)
        ax[2].legend()
        ax[2].set_title("Vertical Projection")

        if opt["outputdir"] is not None:
            plt.savefig(Path(opt.outputdir, _get_crossection_plot_fname(opt.beam, timestamp)))
        if opt["show_plots"]:
            plt.show()
        figlist.append(fig)
    return figlist


def _aux_variables(ax, bsrt_df, plot_dict, opt):

    ax.plot(
        bsrt_df.index, bsrt_df[plot_dict["variable1"]], color="red", label=plot_dict["variable1"]
    )
    ax.legend(loc="upper left")
    ax.set_title(plot_dict["title"])
    _add_kick_lines(ax, opt["kick_df"])

    if plot_dict["variable2"] is not None:
        ax2 = ax.twinx()
        ax2.plot(
            bsrt_df.index,
            bsrt_df[plot_dict["variable2"]],
            color="blue",
            label=plot_dict["variable2"],
        )
        ax2.legend(loc="upper right")


def plot_auxiliary_variables(opt, bsrt_df):
    plot_dicts = [
        {"variable1": "acqCounter", "variable2": None, "title": "acqCounter"},
        {"variable1": "lastAcquiredBunch", "variable2": None, "title": "lastAcquiredBunch"},
        {"variable1": "cameraGainVoltage", "variable2": None, "title": "cameraGainVoltage"},
        {
            "variable1": "opticsResolutionSet1",
            "variable2": "opticsResolutionSet2",
            "title": "opticsResolutionSet",
        },
        {"variable1": "imageCenterSet1", "variable2": "imageCenterSet2", "title": "imageCenter"},
        {"variable1": "betaTwissSet1", "variable2": "betaTwissSet2", "title": "betaTwiss"},
        {"variable1": "imageScaleSet1", "variable2": "imageScaleSet2", "title": "imageScale"},
    ]

    fig, ax = plt.subplots(nrows=len(plot_dicts), ncols=1, figsize=(9, 20), sharex=True)

    [_aux_variables(axis, bsrt_df, plot_dict, opt) for axis, plot_dict in zip(ax, plot_dicts)]

    if opt["outputdir"] is not None:
        plt.savefig(Path(opt.outputdir, _get_auxiliary_var_plot_fname(opt.beam)))
    if opt["show_plots"]:
        plt.show()
    return fig


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()
