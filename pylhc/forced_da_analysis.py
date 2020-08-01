"""
Forced DA Analysis
-------------------

Runs the forced DA analysis, following the procedure described in `CarlierForcedDA2019`_.


.. _CarlierForcedDA2019: https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.22.031002


:module: forced_da_analysis
:author: jdilly, mihofer

"""
import os
from collections import defaultdict, OrderedDict
from contextlib import suppress
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.transforms as mtrans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytimber
from generic_parser.entrypoint_parser import save_options_to_config
from pytimber.pagestore import PageStore
import scipy.odr
import scipy.optimize
import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import get_multi_class, get_instance_faker_meta, TRUE_ITEMS, FALSE_ITEMS
from omc3.optics_measurements import toolbox
from omc3.plotting.utils import style, lines, annotations, colors
from omc3.tune_analysis.bbq_tools import clean_outliers_moving_average
from omc3.utils import logging_tools
from omc3.utils.time_tools import CERNDatetime
from omc3.definitions.formats import get_config_filename
from pandas.plotting import register_matplotlib_converters
from tfs.tools import significant_digits

from pylhc.constants.forced_da_analysis import (bsrt_emittance_key, bws_emittance_key,
                                                column_action, column_bws_norm_emittance, column_emittance,
                                                column_norm_emittance,
                                                err_col, mean_col, rel_col, sigma_col,
                                                header_da, header_da_error, header_nominal_emittance,
                                                header_norm_nominal_emittance,
                                                outfile_emittance, outfile_emittance_bws, outfile_kick, outfile_plot,
                                                HEADER_ENERGY,
                                                HEADER_TIME_AFTER, HEADER_TIME_BEFORE,
                                                HEADER_BSRT_ROLLING_WINDOW, HEADER_BSRT_OUTLIER_LIMIT,
                                                TIME_AFTER_KICK_S, TIME_AROUND_KICKS_MIN, TIME_BEFORE_KICK_S,
                                                PLOT_FILETYPES, INTENSITY, INITIAL_DA_FIT, INTENSITY_AFTER,
                                                INTENSITY_BEFORE,
                                                INTENSITY_KEY, INTENSITY_LOSSES,
                                                ROLLING_AVERAGE_WINDOW, BSRT_EMITTANCE_TO_METER, BWS_DIRECTIONS,
                                                BWS_EMITTANCE_TO_METER, KICKFILE, RESULTS_DIR, YPAD,
                                                MAX_CURVEFIT_FEV, OUTFILE_INTENSITY, OUTLIER_LIMIT,
                                                )
from pylhc.constants.general import get_proton_gamma, get_proton_beta, LHC_NOMINAL_EMITTANCE, TFS_SUFFIX, TIME_COLUMN

LOG = logging_tools.get_logger(__name__)

# Weird Datatypes
path_or_str = get_multi_class(Path, str)
path_or_str_or_df = get_multi_class(Path, str, tfs.TfsDataFrame, pd.DataFrame)
path_or_str_or_db = get_multi_class(Path, str, PageStore)


class BoolOrPathOrDataFrame(metaclass=get_instance_faker_meta(bool, Path, str, tfs.TfsDataFrame, pd.DataFrame)):
    """ A class that behaves like a boolean when possible, otherwise like a Path, string or Dataframe."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser

        if value in TRUE_ITEMS:
            return True

        elif value in FALSE_ITEMS:
            return False

        else:
            try:
                return Path(value)
            except TypeError:
                return value


def get_params():
    return EntryPointParameters(
        kick_directory=dict(
            flags=['-k', "--kickdir"],
            required=True,
            type=get_multi_class(Path, str),
            help="Analysis kick_directory containing kick files."
        ),
        output_directory=dict(
            flags=['-o', "--outdir"],
            type=path_or_str,
            help="Output kick_directory, if not given subfolder in kick kick_directory",
        ),
        energy=dict(
            flags=['-e', "--energy"],
            required=True,
            type=get_multi_class(float, int),
            help="Beam energy in GeV."
        ),
        fill=dict(
            flags=['-f', "--fill"],
            type=int,
            help="Fill that was used. If not given, check out time_around_kicks."
        ),
        beam=dict(
            flags=["-b", "--beam"],
            required=True,
            choices=[1, 2],
            type=int,
            help="Beam to use."
        ),
        plane=dict(
            flags=["-p", "--plane"],
            choices=["X", "Y", "XY"],
            required=True,
            type=str,
            help="Plane of the kicks. Give 'XY' for using both planes (e.g. diagonal kicks)."
        ),
        time_around_kicks=dict(
            type=int,
            default=TIME_AROUND_KICKS_MIN,
            help=("If no fill is given, this defines the time (in minutes) "
                  "when data before the first and after the last kick is extracted.")
        ),
        intensity_time_before_kick=dict(
            type=int,
            nargs=2,
            default=TIME_BEFORE_KICK_S,
            help=("Defines the times before the kicks (in seconds) "
                  "which is used for intensity averaging to calculate the losses.")
        ),
        intensity_time_after_kick=dict(
            type=int,
            nargs=2,
            default=TIME_AFTER_KICK_S,
            help=("Defines the times after the kicks (in seconds) "
                  "which is used for intensity averaging to calculate the losses.")
        ),
        normalized_emittance=dict(
            type=float,
            default=LHC_NOMINAL_EMITTANCE,
            help=("Assumed NORMALIZED nominal emittance for the machine.")
        ),
        emittance_tfs=dict(
            type=path_or_str_or_df,
            help="Dataframe or Path of pre-saved emittance tfs.",
        ),
        intensity_tfs=dict(
            type=path_or_str_or_df,
            help="Dataframe or Path of pre-saved intensity tfs.",
        ),
        show_wirescan_emittance=dict(
            default=False,
            type=BoolOrPathOrDataFrame,
            help=("Flag if the emittance from wirescan should also be shown, "
                  "can also be a Dataframe or Path of pre-saved emittance bws tfs."),
        ),
        timber_db=dict(
            type=str,
            default='all',
            choices=['all', 'mdb', 'ldb', 'nxcals'],
            help="Which timber database to use."
        ),
        pagestore_db=dict(
            type=path_or_str_or_db,
            help="(Path to-) presaved timber database"
        ),
        fit=dict(
            type=str,
            default='exponential',
            choices=['exponential', 'linear'],
            help="Fitting function to use (rearranges parameters to make sense)."
        ),
        emittance_window_length=dict(
            help="Length of the moving average window. (# data points)",
            type=int,
            default=ROLLING_AVERAGE_WINDOW,
        ),
        emittance_outlier_limit=dict(
            help="Limit, i.e. cut from mean, on emittance outliers in meter.",
            type=float,
            default=OUTLIER_LIMIT,
        ),
        emittance_type=dict(
            type=str,
            default='average',
            choices=['fit_sigma', 'average'],
            help="Which BSRT data to use (from database)."
        ),
        show=dict(
            action='store_true',
            help="Show plots.",
        )
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.debug("Starting Forced DA analysis.")
    _save_options(opt)
    _log_opt(opt)

    # get data
    kick_dir, out_dir = _get_output_dir(opt.kick_directory, opt.output_directory)
    kick_df = _get_kick_df(kick_dir, opt.plane)
    intensity_df, emittance_df, emittance_bws_df = _get_dataframes(kick_df.index,
                                                                   opt.get_subdict(['fill', 'beam', 'plane',
                                                                                    'time_around_kicks',
                                                                                    'emittance_tfs', 'intensity_tfs',
                                                                                    'show_wirescan_emittance',
                                                                                    'timber_db', 'pagestore_db',
                                                                                    'emittance_window_length',
                                                                                    'emittance_outlier_limit',
                                                                                    'emittance_type',
                                                                                    'normalized_emittance']))
    _check_all_times_in(kick_df.index, intensity_df.index[0], intensity_df.index[-1])

    # add data to kicks
    kick_df = _add_intensity_and_losses_to_kicks(kick_df, intensity_df,
                                                 opt.intensity_time_before_kick, opt.intensity_time_after_kick)
    kick_df = _add_emittance_to_kicks(opt.plane, opt.energy, kick_df, emittance_df, opt.normalized_emittance)
    kick_df = _do_fit(opt.plane, kick_df, opt.fit)
    kick_df = _convert_to_sigmas(opt.plane, kick_df)

    # output
    _write_tfs(out_dir, opt.plane, kick_df, intensity_df, emittance_df, emittance_bws_df)

    # plotting
    figs = dict()
    register_matplotlib_converters()  # for datetime plotting
    figs['emittance'] = _plot_emittances(out_dir, opt.beam, opt.plane, emittance_df, emittance_bws_df, kick_df.index)
    figs['intensity'] = _plot_intensity(out_dir, opt.beam, opt.plane, kick_df, intensity_df)
    for fit_type in ('exponential', 'linear', 'norm'):
        figs[f'da_fit_{fit_type}'] = _plot_da_fit(out_dir, opt.beam, opt.plane, kick_df, fit_type)

    if opt.show:
        plt.show()
    return figs


# Helper ---

def _log_opt(opt):
    LOG.info("Performing ForcedDA Analysis for:")
    if opt.fill is not None:
        LOG.info(f"  Fill: {opt.fill}")
    LOG.info(f"  Energy: {opt.energy} GeV")
    LOG.info(f"  Beam: {opt.beam}")
    LOG.info(f"  Plane: {opt.plane}")
    LOG.info(f"  Analysis Directory: '{opt.kick_directory}'")


def _save_options(opt):
    if opt.output_directory:
        out_path = Path(opt.output_directory)
        out_path.mkdir(exist_ok=True, parents=True)
        save_options_to_config(out_path / get_config_filename(__file__), OrderedDict(sorted(opt.items())))


def _write_tfs(out_dir, plane, kick_df, intensity_df, emittance_df, emittance_bws_df):
    """ Write out gathered data. """
    LOG.debug("Writing tfs files.")
    for df in (kick_df, intensity_df, emittance_df, emittance_bws_df):
        if df is not None:
            df.insert(0, TIME_COLUMN, [CERNDatetime(dt).cern_utc_string() for dt in df.index])
    try:
        tfs.write(out_dir / outfile_kick(plane), kick_df)
        tfs.write(out_dir / OUTFILE_INTENSITY, intensity_df)
        tfs.write(out_dir / outfile_emittance(plane), emittance_df)
        if emittance_bws_df is not None:
            tfs.write(out_dir / outfile_emittance_bws(plane), emittance_bws_df)
    except (FileNotFoundError, IOError):
        LOG.error(f"Cannot write into directory: {str(out_dir)} ")


def _check_all_times_in(series, start, end):
    """ Check if all times in series are between start and end. """
    if any(s for s in series if s < start or s > end):
        raise ValueError("Some of the kick-times are outside of the fill times! "
                         "Check if correct kick-file or fill number are used.")


def _convert_time_index(list_, path=None):
    for index_convert in (_datetime_to_cerntime_index, _string_to_cerntime_index, _timestamp_to_cerntime_index):
        with suppress(TypeError):
            return index_convert(list_)
    msg = f"Unrecognized format in column '{TIME_COLUMN}'"
    if path:
        msg += f" in '{str(path)}'"
    raise TypeError(msg)


def _string_to_cerntime_index(list_):
    return pd.Index((CERNDatetime.from_cern_utc_string(t) for t in list_), dtype=object)


def _timestamp_to_cerntime_index(list_):
    return pd.Index((CERNDatetime.from_timestamp(t) for t in list_), dtype=object)


def _datetime_to_cerntime_index(list_):
    return pd.Index((CERNDatetime(t) for t in list_), dtype=object)


def _drop_duplicate_indices(df):
    duplicate_mask = [True] + [df.index[idx] != df.index[idx-1] for idx in range(1, len(df.index))]
    return df.loc[duplicate_mask, :]


# TFS Data Loading -------------------------------------------------------------


def _get_dataframes(kick_times, opt):
    db = _get_db(opt)

    if opt.fill is not None:
        timespan_ts = _get_fill_times(db, opt.fill)
        timespan_dt = _convert_time_index(timespan_ts)
    else:
        td = pd.Timedelta(minutes=opt.time_around_kicks)
        timespan_dt = (kick_times.min() - td, kick_times.max() + td)
        timespan_ts = tuple(t.timestamp() for t in timespan_dt)

    if opt.intensity_tfs:
        intensity_df = _read_tfs(opt.intensity_tfs, timespan_dt)
    else:
        intensity_df = _get_bctrf_beam_intensity_from_timber(opt.beam, db, timespan_ts)

    if opt.emittance_tfs:
        emittance_df = _read_tfs(opt.emittance_tfs, timespan_dt)
    else:
        emittance_df = _get_bsrt_bunch_emittances_from_timber(opt.beam, opt.plane, db, timespan_ts,
                                                              opt.emittance_type, opt.normalized_emittance)
    emittance_df = _filter_emittance_data(emittance_df, opt.plane, opt.emittance_window_length, opt.emittance_outlier_limit)

    if opt.show_wirescan_emittance is True:
        emittance_bws_df = _get_bws_emittances_from_timber(opt.beam, opt.plane, db, timespan_ts)
    elif opt.show_wirescan_emittance:
        emittance_bws_df = _read_tfs(opt.show_wirescan_emittance, timespan_dt)
    else:
        emittance_bws_df = None

    return intensity_df, emittance_df, emittance_bws_df


def _read_tfs(tfs_file_or_path, timespan):
    """ Read previously gathered data (see :meth:`pylhc.forced_da_analysis._write_tfs`)"""
    try:
        tfs_df = tfs.read_tfs(tfs_file_or_path, index=TIME_COLUMN)
    except IOError:
        tfs_df = tfs_file_or_path  # hopefully

    tfs_df.index = _convert_time_index(tfs_df.index)

    return tfs_df.loc[slice(*timespan), :]


def _filter_emittance_data(df, planes, window_length, limit):
    for plane in planes:
        col_nemittance = column_norm_emittance(plane)
        col_err_nemittance = err_col(col_nemittance)
        col_mean = mean_col(col_nemittance)
        col_err_mean = err_col(col_mean)

        mav, std, mask = clean_outliers_moving_average(df[col_nemittance],
                                                       length=window_length,
                                                       limit=limit)
        df[col_mean] = mav
        df[col_err_mean] = std
        # if any(df[col_err_nemittance]):
        #     df[col_err_mean] = _rolling_errors(df[col_err_nemittance], ~mask, window_length)

    df = df.dropna(axis='index')
    if len(df.index) == 0:
        raise IndexError("Not enough emittance data extracted. Try to give a fill number.")

    df.headers[HEADER_BSRT_ROLLING_WINDOW] = window_length
    df.headers[HEADER_BSRT_OUTLIER_LIMIT] = limit
    df = _maybe_add_sum_for_planes(df, planes, column_norm_emittance)
    df = _maybe_add_sum_for_planes(df, planes,
                                   lambda p: mean_col(column_norm_emittance(p)),
                                   lambda p: err_col(mean_col(column_norm_emittance(p))))
    return df


def _rolling_errors(data_series, clean_mask, length):
    data = data_series.copy()
    data[clean_mask] = np.NaN

    # 'interpolate' fills nan based on index/values of neighbours
    data = data.interpolate("index").fillna(method="bfill").fillna(method="ffill")

    shift = -int((length-1)/2)  # Shift average to middle value

    # calculate sqrt(sum(x**2))/N, fill NaNs at the ends
    return np.sqrt(np.square(data).rolling(length).sum().shift(shift).fillna(
        method="bfill").fillna(method="ffill")) / length


# Timber Data ------------------------------------------------------------------


def _get_db(opt):
    db = None

    if opt.pagestore_db:
        db = opt.pagestore_db
        try:
            db_path = Path(db)
        except TypeError:
            pass
        else:
            db = PageStore(f'file:{str(db_path)}', str(db_path.with_suffix('')))
            if opt.fill is not None:
                raise EnvironmentError("'fill' can't be used with pagestore database.")
    else:
        try:
            db = pytimber.LoggingDB(source=opt['timber_db'])
        except AttributeError:
            pass

    if not db:
        error_msg = ""
        if opt.fill is not None:
            error_msg += "'fill' is given, "
        if opt.emittance_tfs is None:
            error_msg += "'emittance_tfs' is not given, "
        if opt.intensity_tfs is None:
            error_msg += "'intensity_tfs' is not given, "
        if opt.show_wirescan_emittance is True:
            error_msg += 'wirescan emittance is requested, '
        if len(error_msg):
            error_msg += "but there is no database given and no access to timber databases. Aborting."
            raise EnvironmentError(error_msg)
    return db


def _get_fill_times(db, fill):
    LOG.debug(f"Getting Timespan from fill {fill}")
    filldata = db.getLHCFillData(fill)
    return filldata['startTime'], filldata['endTime']


def _get_bctrf_beam_intensity_from_timber(beam, db, timespan):
    LOG.debug(f"Getting beam intensity from bctfr for beam {beam}.")
    intensity_key = INTENSITY_KEY.format(beam=beam)
    LOG.debug(f"  Key: {intensity_key}")
    x, y = db.get(intensity_key, *timespan)[intensity_key]
    df = tfs.TfsDataFrame(data=y, index=_timestamp_to_cerntime_index(x),
                          columns=[INTENSITY], dtype=float)

    df = _drop_duplicate_indices(df)
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _get_bsrt_bunch_emittances_from_timber(beam, planes, db, timespan, key_type, nominal_emittance):
    dfs = {p: None for p in planes}
    for plane in planes:
        LOG.debug(f"Getting emittance from BSRT for beam {beam}  and plane {plane}.")
        bunch_emittance_key = bsrt_emittance_key(beam, plane, key_type)
        LOG.debug(f"  Key: {bunch_emittance_key}")
        col_nemittance = column_norm_emittance(plane)
        all_columns = [f(col_nemittance) for f in (lambda s: s, mean_col, err_col)] + [err_col(mean_col(col_nemittance))]

        x, y = db.get(bunch_emittance_key, *timespan)[bunch_emittance_key]
        y_std = np.zeros_like(x)
        if key_type == "fit_sigma":
            # add all data with the same timestamp
            y_new = defaultdict(list)
            for x_elem, y_elem in zip(x, y):
                y_new[f"{x_elem:.3f}"] += y_elem.tolist()

            # get average and std per timestamp
            x = np.array([float(elem) for elem in y_new.keys()])
            y = np.array([np.average(elem) for elem in y_new.values()]) * nominal_emittance
            y_std = np.array([np.std(elem) for elem in y_new.values()]) * nominal_emittance
        elif key_type == 'average':
            y *= BSRT_EMITTANCE_TO_METER
            y_std *= BSRT_EMITTANCE_TO_METER

        # remove entries with zero emittance as unphysical
        x, y, y_std = x[y != 0], y[y != 0], y_std[y != 0]

        df = tfs.TfsDataFrame(index=_timestamp_to_cerntime_index(x),
                              columns=all_columns, dtype=float,)
        df[col_nemittance] = y
        df[err_col(col_nemittance)] = y_std

        dfs[plane] = df

    df = _merge_df_planes(dfs, planes)
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _get_bws_emittances_from_timber(beam, planes, db, timespan):
    dfs = {p: None for p in planes}
    for plane in planes:
        LOG.debug(f"Getting emittance from BWS for beam {beam} and plane {plane}.")
        all_columns = [column_bws_norm_emittance(plane, d) for d in BWS_DIRECTIONS]
        df = None
        for direction in BWS_DIRECTIONS:
            emittance_key = bws_emittance_key(beam, plane, direction)
            LOG.debug(f"  Key: {emittance_key}")
            column_nemittance = column_bws_norm_emittance(plane, direction)

            x, y = db.get(emittance_key, *timespan)[emittance_key]
            if df is None:
                df = tfs.TfsDataFrame(index=_timestamp_to_cerntime_index(x), columns=all_columns, dtype=float)
            df[column_nemittance] = y * BWS_EMITTANCE_TO_METER
            df[column_nemittance] = df[column_nemittance].apply(np.mean)  # BWS can give multiple values
            df[err_col(column_nemittance)] = df[column_nemittance].apply(np.std)  # BWS can give multiple values
            dfs[plane] = df

    df = _merge_df_planes(dfs, planes)
    for direction in BWS_DIRECTIONS:
        df = _maybe_add_sum_for_planes(df, planes,
                                       lambda p: column_bws_norm_emittance(p, direction),
                                       lambda p: err_col(column_bws_norm_emittance(p, direction))
                                       )
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


# Kick Data --------------------------------------------------------------------


def _get_kick_df(kick_dir, plane):
    def column_action_error(x):
        return err_col(column_action(x))

    try:
        df = _get_new_kick_file(kick_dir, plane)
    except FileNotFoundError:
        LOG.debug("Reading of kickfile failed. Looking for old kickfile.")
        df = _get_old_kick_file(kick_dir, plane)

    df = _maybe_add_sum_for_planes(df, plane, column_action, column_action_error)
    return df[[column_action(plane), column_action_error(plane)]]


def _get_old_kick_file(kick_dir, plane):
    """ Kick files from Beta-Beat.src """
    path = kick_dir / "getkickac.out"
    LOG.debug(f"Reading kickfile '{str(path)}'.'")
    df = tfs.read(path)
    df = df.set_index(TIME_COLUMN)
    df.index = _convert_time_index(df.index, path)
    rename_dict = {}
    for p in plane:
        rename_dict.update({f"2J{p}RES": column_action(p),
                            f"2J{p}STDRES": err_col(column_action(p)),
                            f"J{p}2": column_action(p),  # pre 2017
                            f"J{p}STD": err_col(column_action(p)),  # pre 2017
                            })
    df = df.rename(rename_dict, axis="columns")
    df.loc[:, rename_dict.values()] = df.loc[:, rename_dict.values()] * 1e-6
    return df


def _get_new_kick_file(kick_dir, planes):
    """ Kick files from OMC3 """
    dfs = {p: None for p in planes}
    for plane in planes:
        path = kick_dir / f"{KICKFILE}_{plane.lower()}{TFS_SUFFIX}"
        LOG.debug(f"Reading kickfile '{str(path)}'.'")
        df = tfs.read(path, index=TIME_COLUMN)
        df.index = pd.Index([CERNDatetime.from_cern_utc_string(t) for t in df.index], dtype=object)
        dfs[plane] = df
    return _merge_df_planes(dfs, planes)


def _get_output_dir(kick_directory, output_directory):
    kick_path = Path(kick_directory)
    if output_directory:
        output_path = Path(output_directory)
    else:
        output_path = kick_path / RESULTS_DIR
    try:
        output_path.mkdir(exist_ok=True)
    except PermissionError:
        LOG.warn(f"You have no writing permission in '{str(output_path)}', "
                 f"output data might not be created.")
    return kick_path, output_path


# Intensity at Kicks -----------------------------------------------------------


def _add_intensity_and_losses_to_kicks(kick_df, intensity_df, time_before, time_after):
    LOG.debug("Calculating intensity and losses for the kicks.")
    col_list = [INTENSITY_BEFORE, INTENSITY_AFTER, INTENSITY_LOSSES]
    new_columns = [col for col in col_list + [err_col(c) for c in col_list]]
    kick_df = kick_df.reindex(columns=kick_df.columns.tolist() + new_columns)
    kick_df = _get_intensities_around_kicks(kick_df, intensity_df, time_before, time_after)
    kick_df = _calculate_intensity_losses_at_kicks(kick_df)
    return kick_df


def _get_intensities_around_kicks(kick_df, intensity_df, time_before, time_after):
    LOG.debug("Calculating beam intensity before and after kicks.")
    # input signs and order does not matter
    time_before = sorted(-np.abs(t) for t in time_before)
    time_after = sorted(np.abs(t) for t in time_after)

    kick_df.headers[HEADER_TIME_BEFORE] = str(time_before)
    kick_df.headers[HEADER_TIME_AFTER] = str(time_after)
    for i, time in enumerate(kick_df.index):
        # calculate intensity before and after kicks (with error)
        for column, time_delta in ((INTENSITY_BEFORE, time_before), (INTENSITY_AFTER, time_after)):
            t_from, t_to = time + pd.Timedelta(seconds=time_delta[0]), time + pd.Timedelta(seconds=time_delta[1]),
            data = intensity_df.loc[t_from:t_to, INTENSITY]  # awesome pandas can handle time intervals!
            kick_df.loc[time, [column, err_col(column)]] = data.mean(), data.std()
    return kick_df


def _calculate_intensity_losses_at_kicks(kick_df):
    LOG.debug("Calculating intensity losses.")
    # absolute losses
    kick_df[INTENSITY_LOSSES] = kick_df[INTENSITY_BEFORE] - kick_df[INTENSITY_AFTER]
    kick_df[err_col(INTENSITY_LOSSES)] = np.sqrt(
            np.square(kick_df[err_col(INTENSITY_BEFORE)]) + np.square(kick_df[err_col(INTENSITY_AFTER)]))

    # relative losses, error from error-propagation formular for losses / I_before = 1 - I_after / I_before
    kick_df[rel_col(INTENSITY_LOSSES)] = kick_df[INTENSITY_LOSSES] / kick_df[INTENSITY_BEFORE]
    kick_df[rel_col(err_col(INTENSITY_LOSSES))] = np.sqrt(
        np.square(kick_df[INTENSITY_AFTER]/kick_df[INTENSITY_BEFORE]) *
        (np.square(kick_df[err_col(INTENSITY_AFTER)]/kick_df[INTENSITY_AFTER]) +
         np.square(kick_df[err_col(INTENSITY_BEFORE)] / kick_df[INTENSITY_BEFORE]))
    )
    return kick_df


# Emittance at Kicks -----------------------------------------------------------


def _add_emittance_to_kicks(plane, energy, kick_df, emittance_df, nominal):
    LOG.debug("Retrieving normalized emittance at the kicks.")
    kick_df.headers[HEADER_ENERGY] = energy
    kick_df.headers[HEADER_BSRT_ROLLING_WINDOW] = ROLLING_AVERAGE_WINDOW
    col_nemittance = column_norm_emittance(plane)
    cols_emitt = [mean_col(col_nemittance), err_col(mean_col(col_nemittance))]
    cols_kick = [col_nemittance, err_col(col_nemittance)]

    kick_df = kick_df.reindex(columns=kick_df.columns.tolist() + cols_kick)
    idx_emitt = [emittance_df.columns.get_loc(c) for c in cols_emitt]
    for time in kick_df.index:
        idx_kick = emittance_df.index.get_loc(time, method="nearest")
        kick_df.loc[time, cols_kick] = emittance_df.iloc[idx_kick, idx_emitt].values

    # add de-normalized emittance
    normalization = get_proton_gamma(energy) * get_proton_beta(energy)  # norm emittance to emittance
    col_emittance = column_emittance(plane)

    kick_df.headers[header_norm_nominal_emittance(plane)] = nominal
    kick_df.headers[header_nominal_emittance(plane)] = nominal / normalization
    kick_df[col_emittance] = kick_df[col_nemittance] / normalization
    kick_df[err_col(col_emittance)] = kick_df[err_col(col_nemittance)] / normalization
    return kick_df


# Forced DA Fitting ------------------------------------------------------------


def fun_exp_decay(p, x):
    """ p = DA_J, x[0] = action (2J res), x[1] = emittance"""
    return np.exp(-(p - (0.5*x[0])) / x[1])


def fun_exp_sigma(p, x):
    """ p = DA_sigma, x = action (J_sigma) """
    return np.exp(-0.5*(p**2 - x**2))


def fun_linear(p, x):
    """ p = DA_J, x = action (2J res)"""
    return x*0.5 - p


def swap_fun_parameters(fun):
    """ Parameter swapped for Curvefit """
    return lambda x, p: fun(p, x)


def _do_fit(plane, kick_df, fit_type):
    LOG.debug("Fitting forced da to exponential. ")
    action, emittance, rel_losses = _get_fit_data(kick_df, plane)
    init_guess = [INITIAL_DA_FIT*kick_df.headers[header_nominal_emittance(plane)]]

    get_fit_param = {'linear': _linear_fit_parameters,
                     'exponential': _exponential_fit_parameters,
                     }[fit_type]

    fit_fun, x, y, sx, sy = get_fit_param(action, emittance, rel_losses)

    # do prelim fit
    init_fit, _ = _fit_curve(swap_fun_parameters(fit_fun), x, y, init_guess)

    # do odr
    odr = _fit_odr(fit_fun, x, y, sx, sy, init_fit)

    # add DA to kick
    da = odr.beta[0], odr.sd_beta[0]
    kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)] = da
    LOG.info(f"Forced DA (wrt. J) in {plane} [m]: {da[0]} ± {da[1]}")

    return kick_df


def _get_fit_data(kick_df, plane):
    col_action = column_action(plane)
    col_emittance = column_emittance(plane)
    col_losses = rel_col(INTENSITY_LOSSES)

    # get data
    action = kick_df[col_action], _no_nonzero_errors(kick_df[err_col(col_action)])
    emittance = kick_df[col_emittance], _no_nonzero_errors(kick_df[err_col(col_emittance)])
    rel_losses = kick_df[col_losses], _no_nonzero_errors(kick_df[err_col(col_losses)])
    return action, emittance, rel_losses


def _exponential_fit_parameters(action, emittance, rel_losses):
    x = action[0], emittance[0]
    y = rel_losses[0]

    sx = [action[1], emittance[1]]
    sy = rel_losses[1]
    return fun_exp_decay, x, y, sx, sy


def _linear_fit_parameters(action, emittance, rel_losses):
    log_losses = np.log(rel_losses[0])
    x = action[0]
    y = emittance[0] * log_losses

    sx = action[1]
    sy = np.sqrt((log_losses * emittance[1])**2 +
                 ((emittance[0] * rel_losses[1])/rel_losses[0])**2)
    return fun_linear, x, y, sx, sy


def _fit_curve(fun, x, y, init):
    """ Preliminary curve fit, without errors. """
    fit, cov = scipy.optimize.curve_fit(fun, x, y,
                                        p0=init, maxfev=MAX_CURVEFIT_FEV)
    LOG.info(f"Initial DA fit: {fit} with cov {cov}")
    return fit, np.sqrt(np.diag(cov))


def _fit_odr(fun, x, y, sx, sy, init):
    """ ODR Fit """
    # fill zero errors with the minimum error - otherwise fit will not work
    fit_model_sigma = scipy.odr.Model(fun)
    data_model_sigma = scipy.odr.RealData(
        x=x, y=y, sx=sx, sy=sy,
    )
    da_odr = scipy.odr.ODR(data_model_sigma, fit_model_sigma,
                           beta0=init)
    # da_odr.set_job(fit_type=2)
    odr_output = da_odr.run()
    logging_tools.odr_pprint(LOG.info, odr_output)
    return odr_output


def _no_nonzero_errors(series):
    series = series.copy()
    series[series == 0] = np.abs(series[series != 0]).min()
    return series


def _convert_to_sigmas(plane, kick_df):
    """ Converts the DA and the Action into Sigma-Units. """
    LOG.debug("Calculating action and da in sigmas.")
    nominal_emittance = kick_df.headers[header_nominal_emittance(plane)]
    emittance = kick_df[column_emittance(plane)]
    emittance_mean, emittance_std = emittance.mean(), emittance.std()
    emittance_sign, emittance_sign_std = significant_digits(emittance_mean*1e12, emittance_std*1e12)
    LOG.info(f"Measured Emittance {emittance_sign} ± {emittance_sign_std} pm"
             f" (Nominal {nominal_emittance*1e12: .2f} pm)")

    # DA (in units of J) to DA_sigma
    da, da_err = kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)]
    # da_sigma, da_sigma_err = np.sqrt(2*da/nominal_emittance), da_err / np.sqrt(2 * da * nominal_emittance)
    da_sigma, da_sigma_err = np.sqrt(2*da/emittance_mean), da_err / np.sqrt(2 * da * emittance_mean)
    kick_df.headers[header_da(plane, unit="sigma")] = da_sigma
    kick_df.headers[header_da_error(plane, unit="sigma")] = da_sigma_err
    LOG.info(f"Forced DA {plane} in N-sigma: {da_sigma} ± {da_sigma_err}")

    # Action (in units of 2J) to J_sigma
    col_action = column_action(plane)
    # kick_df[sigma_col(col_action)] = np.sqrt(kick_df[col_action] / nominal_emittance)
    # kick_df[err_col(sigma_col(col_action))] = (
    #         0.5 * kick_df[err_col(col_action)] / np.sqrt(kick_df[col_action] * nominal_emittance)
    # )
    kick_df[sigma_col(col_action)] = np.sqrt(kick_df[col_action] / emittance)
    kick_df[err_col(sigma_col(col_action))] = (
            0.5 * kick_df[err_col(col_action)] / np.sqrt(kick_df[col_action] * emittance)
    )
    return kick_df


# Plotting ---------------------------------------------------------------------


def _plot_intensity(directory, beam, plane, kick_df, intensity_df):
    """ Plots beam intensity. For losses the absolute values are used and then normalized
    to the Intensity before the kicks, to get the percentage relative to that (global) value."""
    LOG.debug("Plotting beam intensity")
    style.set_style("standard")
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16.80, 7.68))

    x_span = (kick_df.index.max() - kick_df.index.min()).seconds * np.array([0.03, 0.09])  # defines x-limits

    # convert to % relative to before first kick
    idx_before = intensity_df.index.get_loc(kick_df.index.min() - pd.Timedelta(seconds=x_span[0]), method="ffill")
    idx_intensity = intensity_df.columns.get_loc(INTENSITY)  # for iloc
    intensity_start = intensity_df.iloc[idx_before, idx_intensity]
    norm = intensity_start/100.

    # plot intensity
    ax.plot(mdates.date2num(intensity_df.index),
            intensity_df[INTENSITY] / norm,
            marker=".",
            markersize=mpl.rcParams["lines.markersize"] * 0.5,
            fillstyle="full",
            color=colors.get_mpl_color(0),
            label=f'Intensity')

    # plot losses per kick
    normalized_intensity = kick_df.loc[:, [INTENSITY_BEFORE, INTENSITY_AFTER]]/norm
    normalized_intensity_error = kick_df.loc[:, [err_col(INTENSITY_BEFORE), err_col(INTENSITY_AFTER)]]/norm
    normalized_losses = kick_df.loc[:, [INTENSITY_LOSSES, err_col(INTENSITY_LOSSES)]]/norm
    normalized_losses_kick = kick_df.loc[:, [rel_col(INTENSITY_LOSSES), err_col(rel_col(INTENSITY_LOSSES))]]*100

    for idx, kick in enumerate(kick_df.index):
        ax.errorbar([mdates.date2num(kick)] * 2, normalized_intensity.loc[kick, :],
                    yerr=normalized_intensity_error.loc[kick, :],
                    color=colors.get_mpl_color(1),
                    marker='.',
                    linestyle="-",
                    label='__nolegend__' if idx > 0 else "Losses")

        ax.text(mdates.date2num(kick), .5 * sum(normalized_intensity.loc[kick, :]),
                "  -{:.1f}$\pm${:.1f} %\n".format(*normalized_losses.loc[kick, :]) +
                " (-{:.1f}$\pm${:.1f} %)".format(*normalized_losses_kick.loc[kick, :]),
                va="bottom", color=colors.get_mpl_color(1),
                fontdict=dict(fontsize=mpl.rcParams["font.size"] * 0.8)
                )

    _plot_kicks_and_scale_x(ax, kick_df.index, pad=x_span)
    ylim = [normalized_intensity.min().min(), normalized_intensity.max().max()]
    ypad = 0.1 * (ylim[1]-ylim[0])
    ax.set_ylim([ylim[0]-ypad, ylim[1]+ypad])
    ax.set_ylabel(r'Beam Intensity [%]')
    annotations.make_top_legend(ax, ncol=3)
    plt.tight_layout()
    annotations.set_name(f"Intensity Beam {beam}, Plane {plane}", fig)
    annotations.set_annotation(
        f'Intensity at 100%: {intensity_start*1e-10:.3f}'
        '$\;\cdot\;10^{{10}}$ charges',
        ax=ax, position='left'
    )
    _save_fig(directory, plane, fig, 'intensity')
    return fig


def _plot_emittances(directory, beam, plane, emittance_df, emittance_bws_df, kick_times):
    LOG.debug("Plotting normalized emittances")
    style.set_style("standard")
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.24, 7.68))

    col_norm_emittance = column_norm_emittance(plane)

    bsrt_color = colors.get_mpl_color(0)
    bws_color = colors.get_mpl_color(1)

    ax.errorbar(mdates.date2num(emittance_df.index),
                emittance_df[col_norm_emittance] * 1e6,  # Actual BSRT measurement
                yerr=emittance_df[err_col(col_norm_emittance)] * 1e6,
                color=bsrt_color,
                marker='o',
                markeredgewidth=2,
                linestyle='None',
                label=f'From BSRT')

    ax.errorbar(mdates.date2num(emittance_df.index),
                emittance_df[mean_col(col_norm_emittance)] * 1e6,
                yerr=emittance_df[err_col(mean_col(col_norm_emittance))] * 1e6,
                color=colors.change_color_brightness(bsrt_color, 0.7),
                marker='',
                label=f'Moving Average (window = {ROLLING_AVERAGE_WINDOW})')

    if emittance_bws_df is not None and len(emittance_bws_df.index):
        for d in BWS_DIRECTIONS:
            label = "__nolegend__" if d == BWS_DIRECTIONS[1] else f"From BWS"
            color = bws_color if d == BWS_DIRECTIONS[1] else colors.change_color_brightness(bws_color, 0.5)
            col_bws_nemittance = column_bws_norm_emittance(plane, d)
            ax.errorbar(
                mdates.date2num(emittance_bws_df.index),
                emittance_bws_df[col_bws_nemittance] * 1e6,
                yerr=emittance_bws_df[err_col(col_bws_nemittance)] * 1e6,
                linestyle='None', marker='o', color=color, label=label,
                markersize=mpl.rcParams['lines.markersize']*1.5,
            )

    _plot_kicks_and_scale_x(ax, kick_times)
    ax.set_ylabel(r'$\epsilon_{n}$ $[\mu m]$')
    annotations.make_top_legend(ax, ncol=2)
    plt.tight_layout()
    annotations.set_name(f"Emittance Beam {beam}, Plane {plane}", fig)
    _save_fig(directory, plane, fig, 'emittance')
    return fig


def _plot_da_fit(directory, beam, plane, k_df, fit_type):
    """ Plot the Forced Dynamic Aperture fit.
    (I do not like the complexity of this function. jdilly)"""
    LOG.debug(f"Plotting Dynamic Aperture Fit for {fit_type}")
    style.set_style("standard")
    col_action = column_action(plane)
    col_action_sigma = sigma_col(col_action)
    col_emittance = column_emittance(plane)
    col_intensity = rel_col(INTENSITY_LOSSES)

    kick_df = k_df.copy()
    kick_df = kick_df.sort_values(by=col_action)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.24, 7.68))

    # Plot Measurement Data
    intensity = kick_df[col_intensity]
    intensity_err = kick_df[err_col(col_intensity)]
    if fit_type == 'linear':
        intensity_err = np.abs(1/intensity) * intensity_err
        intensity = np.log(intensity)
    else:
        intensity *= 100
        intensity_err *= 100

    if fit_type == 'norm':
        action = kick_df[col_action_sigma]
        action_err = kick_df[err_col(col_action_sigma)]
        action_x = action
        action_xerr = action_err
    else:
        action = kick_df[col_action]
        action_err = kick_df[err_col(col_action)]
        action_x = action * 1e6
        action_xerr = action_err * 1e6

    ax.errorbar(
        action_x,
        intensity,
        xerr=action_xerr,
        yerr=intensity_err,
        marker=".",
        color=colors.get_mpl_color(0),
        label=f'Kicks'
    )

    # Plot Fit
    emittance = kick_df[col_emittance]
    da, da_err = kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)]
    da_mu, da_err_mu = significant_digits(da*1e6, da_err*1e6)
    da_label = f'Fit: DA$_J$= ${da_mu} \pm {da_err_mu} \mu m$'

    if fit_type == 'linear':
        fit_fun = fun_linear
        fit_data = action
        multiplier = 1/emittance  # DA-J/emittance = -ln(I/Io)
    elif fit_type == 'exponential':
        fit_fun = fun_exp_decay
        fit_data = (action, emittance)
        multiplier = 100  # for percentages
    elif fit_type == 'norm':
        da, da_err = (kick_df.headers[header_da(plane, unit="sigma")],
                      kick_df.headers[header_da_error(plane, unit="sigma")])
        da_round, da_err_round = significant_digits(da, da_err)
        da_label = f'Fit: DA= ${da_round} \pm {da_err_round} N_{{\sigma}}$'
        fit_fun = fun_exp_sigma
        fit_data = action
        multiplier = 100  # for percentages

    fit_mean = fit_fun(da, fit_data) * multiplier
    fit_min = fit_fun(da - da_err, fit_data) * multiplier
    fit_max = fit_fun(da + da_err, fit_data) * multiplier

    color = colors.get_mpl_color(1)
    ax.fill_between(action_x, fit_min, fit_max,
                    facecolor=mcolors.to_rgba(color, .3))

    ax.plot(action_x, fit_mean, ls="--", c=color, label=da_label)

    # extend fit to 100% losses
    color_ext = '#7f7f7f'
    action_max = action.max()
    emittance_at_max = emittance[action == action_max][0]

    if fit_type in ['linear', 'exponential']:
        da_x = da * 2*1e6
        da_string = "2DA$_J$"
    elif fit_type == 'norm':
        da_x = da
        da_string = "DA$_\sigma$"

    if action_max < da:
        if fit_type in ['linear', 'exponential']:
            action_ext = np.linspace(action_max, 2*da, 10)
            action_x_ext = action_ext * 1e6
            if fit_type == 'exponential':
                fit_data_ext = (action_ext, emittance_at_max)
            elif fit_type == 'linear':
                fit_data_ext = action_ext
                multiplier = 1/emittance_at_max
        else:
            action_ext = np.linspace(action_max, da, 10)
            action_x_ext = action_ext
            fit_data_ext = action_ext

        fit_ext = fit_fun(da, fit_data_ext) * multiplier
        ax.plot(action_x_ext, fit_ext, ls='--', color=mcolors.to_rgba(color_ext, .3), label='__nolegend__')
    # DA Marker
    ax.axvline(da_x, ls='--', color=color_ext, marker='', label='__nolegend__')
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)  # x is data, y is axes
    ax.text(x=da_x, y=1.0, s=da_string, va='bottom', ha="center", zorder=-1, color=color_ext, transform=trans)

    # Format figure
    if fit_type == 'norm':
        nominal_emittance = kick_df.headers[header_nominal_emittance(plane)]
        emittance_mean, emittance_std = emittance.mean(), emittance.std()
        emittance_sign, emittance_sign_std = significant_digits(emittance_mean*1e12, emittance_std*1e12)
        ax.text(
            x=0, y=1.00,
            s=(f"$\epsilon_{{mean}}$ = {emittance_sign} $\pm$ {emittance_sign_std} pm "
               f"($\epsilon_{{nominal}}$ = {nominal_emittance*1e12: .2f} pm)"),
            transform=ax.transAxes,
            va='bottom', ha='left',
        )
        ax.set_xlabel(f"$N_{{\sigma}} = \sqrt{{2J_{{{plane if len(plane) == 1 else ''}}}/\epsilon}}$")
    else:
        ax.set_xlabel(f"$2J_{{{plane if len(plane) == 1 else ''}}} \; [\mu m]$")

    if fit_type == "linear":
        ax.set_ylabel(r'ln($I/I_0$)')
    else:
        ax.set_ylabel(r'Beam Losses [%]')
        ax.set_ylim([0, intensity.max() * (1 + YPAD)])
    ax.set_xlim([0, None])
    annotations.make_top_legend(ax, ncol=3)
    plt.tight_layout()
    annotations.set_name(f"DA {'' if fit_type == 'norm' else 'J'} {fit_type} Fit {beam}, Plane {plane}", fig)
    _save_fig(directory, plane, fig, f'dafit_{fit_type}')
    return fig


def _get_fit_plot_data(da, da_err, data, fit_type):
    fit_fun = {'exponential': fun_exp_decay, 'linear': fun_linear}[fit_type]
    multiplier = 100  # for percentages
    if fit_type == 'linear':
        multiplier = 1/data[1]  # DA-J/emittance = -ln(I/Io)
        data = data[0]
    fit_mean = fit_fun(da, data) * multiplier
    fit_min = fit_fun(da - da_err, data) * multiplier
    fit_max = fit_fun(da + da_err, data) * multiplier
    return fit_mean, fit_min, fit_max


# Helper ---


def _plot_kicks_and_scale_x(ax, kick_times, pad=20):
    try:
        lines.plot_vertical_lines_fast(ax, kick_times, color='grey', linestyle='--', alpha=0.8, marker='', label="Kicks")
    except AttributeError:
        # old version, before plot_optics_measurements (remove in future, jdilly 30.07.2020)
        lines.vertical_lines(ax, mdates.date2num(kick_times), color='grey', linestyle='--', alpha=0.8, marker='', label="Kicks")

    first_kick, last_kick = kick_times.min(), kick_times.max()
    try:
        time_delta = [pd.Timedelta(seconds=pad[i]) for i in range(2)]
    except TypeError:
        time_delta = [pd.Timedelta(seconds=pad) for _ in range(2)]

    # ax.set_xlim([(first_kick - time_delta[0]).timestamp, last_kick + time_delta[1]])  # worked in the past
    ax.set_xlim([mdates.date2num(first_kick - time_delta[0]),
                 mdates.date2num(last_kick + time_delta[1])])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.set_xlabel('Time')
    annotations.set_annotation(f"Date: {first_kick.strftime('%Y-%m-%d')}", ax, "left")


def _merge_df_planes(df_dict, planes):
    """ In case planes == 'XY' merge the df_dict into one dataframe. """
    if len(planes) == 1:
        return df_dict[planes]
    return pd.merge(*df_dict.values(), how='inner', left_index=True, right_index=True)


def _maybe_add_sum_for_planes(df, planes, col_fun, col_err_fun=None):
    """ In case planes == 'XY' add the two plane columns and their errors """
    if len(planes) > 1:
        if col_err_fun is not None:
            cols = lambda p: [col_fun(p), col_err_fun(p)]
            x_cols, y_cols = [cols(p) for p in planes]
            df = df.reindex(columns=df.columns.to_list() + cols(planes))
            df[cols(planes)] = np.array(toolbox.df_sum_with_err(df, a_col=x_cols[0], b_col=y_cols[0],
                                                                a_err_col=x_cols[1], b_err_col=y_cols[1])).T
        else:
            x_col, y_col = [col_fun(p) for p in planes]
            df[col_fun(planes)] = toolbox.df_sum(df, a_col=x_col, b_col=y_col)
    return df


def _save_fig(directory, plane, fig, ptype):
    try:
        for ftype in PLOT_FILETYPES:
            path = os.path.join(directory, outfile_plot(ptype, plane, ftype))
            LOG.debug(f"Saving Figure to {path}")
            fig.savefig(path)
    except IOError:
        LOG.error(f"Couldn't create output files for {ptype} plots.")


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
