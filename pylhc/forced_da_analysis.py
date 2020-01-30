"""
Forced DA Analysis
-------------------

Runs the forced DA analysis, following the procedure described in `CarlierForcedDA2019`_.

*--Required--*

- **beam** *(int)*: Beam to use.

  Flags: **['-b', '--beam']**
  Choices: ``[1, 2]``
- **directory** *(str)*: Analysis directory containing kick files.

  Flags: **['-d', '--dir']**
- **energy** *(float)*: Beam energy in GeV.

  Flags: **['-e', '--energy']**

*--Optional--*

- **fill** *(int)*: Fill that was used. If not given, check out time_around_kicks.

  Flags: **['-f', '--fill']**
- **intensity_time_after_kick** *(int)*: Defines the times after the kicks (in seconds) which is used for intensity averaging to calculate the losses.

  Flags: **['--tafter']**
  Default: ``[5, 30]``
- **intensity_time_before_kick** *(int)*: Defines the times before the kicks (in seconds) which is used for intensity averaging to calculate the losses.

  Flags: **['--tbefore']**
  Default: ``[30, 5]``
- **plane** *(str)*: Plane of the kicks. Give 'XY' for using both planes (e.g. diagonal kicks).

  Flags: **['-p', '--plane']**
  Choices: ``['X', 'Y', 'XY']``
  Default: ``XY``
- **time_around_kicks** *(int)*: If no fill is given, this defines the time (in minutes) when data before the first and after the last kick is extracted.

  Flags: **['--taround']**
  Default: ``10``


.. _CarlierForcedDA2019: https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.22.031002


:module: forced_da_analysis
:author: jdilly

"""


import os
from contextlib import suppress

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytimber
import scipy.odr
import scipy.optimize
import tfs
from generic_parser import EntryPointParameters, entrypoint
from tfs.tools import significant_digits

from pylhc.constants.forced_da_analysis import (bsrt_emittance_key, bws_emittance_key,
                                                column_action, column_bws_norm_emittance, column_emittance,
                                                column_norm_emittance,
                                                err_col, mean_col, rel_col, sigma_col,
                                                header_da, header_da_error, header_nominal_emittance,
                                                header_norm_nominal_emittance,
                                                outfile_emittance, outfile_emittance_bws, outfile_kick, outfile_plot,
                                                TFS_SUFFIX, HEADER_EMITTANCE_AVERAGE, HEADER_ENERGY, HEADER_TIME_AFTER,
                                                HEADER_TIME_BEFORE,
                                                TIME, TIME_AFTER_KICK_S, TIME_AROUND_KICKS_MIN, TIME_BEFORE_KICK_S,
                                                PLOT_FILETYPES, INTENSITY, INITIAL_DA_FIT, INTENSITY_AFTER,
                                                INTENSITY_BEFORE,
                                                INTENSITY_KEY, INTENSITY_LOSSES,
                                                ROLLING_AVERAGE_WINDOW, BSRT_EMITTANCE_TO_METER, BWS_DIRECTIONS,
                                                BWS_EMITTANCE_TO_METER, KICKFILE, RESULTS_DIR, YPAD,
                                                )
from pylhc.constants.general import get_proton_gamma, get_proton_beta, LHC_NOMINAL_EMITTANCE

from omc3.optics_measurements import toolbox
from omc3.utils import logging_tools
from omc3.utils import plot_style as style
from omc3.utils.time_tools import CERNDatetime
from pylhc.plotshop import lines, annotations, colors

LOG = logging_tools.get_logger(__name__)


def get_params():
    return EntryPointParameters(
        directory=dict(
            flags=['-d', "--dir"],
            required=True,
            type=str,
            help="Analysis directory containing kick files."
        ),
        energy=dict(
            flags=['-e', "--energy"],
            required=True,
            type=float,
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
            default="XY",
            type=str,
            help="Plane of the kicks. Give 'XY' for using both planes (e.g. diagonal kicks)."
        ),
        time_around_kicks=dict(
            flags=["--taround"],
            type=int,
            default=TIME_AROUND_KICKS_MIN,
            help=("If no fill is given, this defines the time (in minutes) "
                  "when data before the first and after the last kick is extracted.")
        ),
        intensity_time_before_kick=dict(
            flags=["--tbefore"],
            type=int,
            nargs=2,
            default=TIME_BEFORE_KICK_S,
            help=("Defines the times before the kicks (in seconds) "
                  "which is used for intensity averaging to calculate the losses.")
        ),
        intensity_time_after_kick=dict(
            flags=["--tafter"],
            type=int,
            nargs=2,
            default=TIME_AFTER_KICK_S,
            help=("Defines the times after the kicks (in seconds) "
                  "which is used for intensity averaging to calculate the losses.")
        ),
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.debug("Starting Forced DA analysis.")
    _log_opt(opt)

    # get data
    out_dir = _get_output_dir(opt.directory)
    kick_df = _get_kick_df(opt.directory, opt.plane)
    intensity_df, emittance_df, emittance_bws_df = _get_dfs_from_timber(opt.fill, opt.beam, opt.plane,
                                                                        kick_df.index, opt.time_around_kicks)
    _check_all_times_in(kick_df.index, intensity_df.index[0], intensity_df.index[-1])

    # add data to kicks
    kick_df = _add_intensity_and_losses_to_kicks(kick_df, intensity_df,
                                                 opt.intensity_time_before_kick, opt.intensity_time_after_kick)
    kick_df = _add_emittance_to_kicks(opt.plane, opt.energy, kick_df, emittance_df)
    kick_df = _fit_exponential(opt.plane, kick_df)
    kick_df = _convert_to_sigmas(opt.plane, kick_df)

    # output
    _write_tfs(out_dir, opt.plane, kick_df, emittance_df, emittance_bws_df)

    # plotting
    _plot_emittances(out_dir, opt.beam, opt.plane, emittance_df, emittance_bws_df, kick_df.index)
    _plot_intensity(out_dir, opt.beam, opt.plane, kick_df, intensity_df)
    # _plot_losses(out_dir, beam, plane, kick_df)
    _plot_da_fit(out_dir, opt.beam, opt.plane, kick_df)
    _plot_da_fit_normalized(out_dir, opt.beam, opt.plane, kick_df)

    plt.show()


# Helper ---

def _log_opt(opt):
    LOG.info("Performing ForcedDA Analysis for:")
    if opt.fill is not None:
        LOG.info(f"  Fill: {opt.fill}")
    LOG.info(f"  Energy: {opt.energy} GeV")
    LOG.info(f"  Beam: {opt.beam}")
    LOG.info(f"  Plane: {opt.plane}")
    LOG.info(f"  Analysis Directory: '{opt.directory}'")


def _write_tfs(directory, plane, kick_df, emittance_df, emittance_bws_df):
    """ Write out gathered data. """
    LOG.debug("Writing tfs files.")
    for df in (kick_df, emittance_df, emittance_bws_df):
        df.insert(0, TIME, [CERNDatetime(dt).cern_utc_string() for dt in df.index])
    try:
        tfs.write(os.path.join(directory, outfile_kick(plane)), kick_df)
        tfs.write(os.path.join(directory, outfile_emittance(plane)), emittance_df)
        tfs.write(os.path.join(directory, outfile_emittance_bws(plane)), emittance_bws_df)
    except (FileNotFoundError, IOError):
        LOG.error(f"Cannot write into directory: {directory} ")


def _check_all_times_in(series, start, end):
    """ Check if all times in series are between start and end. """
    if any(s for s in series if s < start or s > end):
        raise ValueError("Some of the kick-times are outside of the fill times! "
                         "Check if correct kick-file or fill number are used.")


# Timber Data ------------------------------------------------------------------


def _get_dfs_from_timber(fill, beam, plane, kick_times, time_around_kicks):
    LOG.debug("Getting data from timber.")
    db = pytimber.LoggingDB()
    if fill is not None:
        LOG.debug("Getting Timespan from fill")
        filldata = db.getLHCFillData(fill)
        timespan = filldata['startTime'], filldata['endTime']
    else:
        td = pd.Timedelta(minutes=time_around_kicks)
        timespan = (kick_times.min() - td, kick_times.max() + td)
        timespan = tuple(t.timestamp() for t in timespan)

    intensity_df = _get_bctrf_beam_intensity(beam, db, timespan)
    emittance_bws_df = _get_bws_emittances(beam, plane, db, timespan)
    emittance_df = _get_bsrt_bunch_emittances(beam, plane,  db, timespan)
    return intensity_df, emittance_df, emittance_bws_df


def _get_bctrf_beam_intensity(beam, db, timespan):
    LOG.debug(f"Getting beam intensity from bctfr for beam {beam}.")
    intensity_key = INTENSITY_KEY.format(beam=beam)
    LOG.debug(f"  Key: {intensity_key}")
    x, y = db.get(intensity_key, *timespan)[intensity_key]
    time_index = pd.Index(CERNDatetime.from_timestamp(t) for t in x)
    df = tfs.TfsDataFrame(data=y, index=time_index, columns=[INTENSITY], dtype=float)
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _get_bsrt_bunch_emittances(beam, planes, db, timespan):
    dfs = {p: None for p in planes}
    for plane in planes:
        LOG.debug(f"Getting emittance from BSRT for beam {beam}  and plane {plane}.")
        bunch_emittance_key = bsrt_emittance_key(beam, plane)
        LOG.debug(f"  Key: {bunch_emittance_key}")
        col_nemittance = column_norm_emittance(plane)
        all_columns = [f(col_nemittance) for f in (lambda s: s, mean_col, err_col)]

        x, y = db.get(bunch_emittance_key, *timespan)[bunch_emittance_key]

        # remove entries with zero emittance as unphysical
        x, y = x[y != 0], y[y != 0]

        time_index = pd.Index(CERNDatetime.from_timestamp(t) for t in x)
        df = tfs.TfsDataFrame(index=time_index,
                              columns=all_columns, dtype=float,
                              headers={HEADER_EMITTANCE_AVERAGE: ROLLING_AVERAGE_WINDOW})
        df[col_nemittance] = y * BSRT_EMITTANCE_TO_METER

        rolling = df[col_nemittance].rolling(window=ROLLING_AVERAGE_WINDOW, center=True)
        df[mean_col(col_nemittance)] = rolling.mean()
        df[err_col(col_nemittance)] = rolling.std()
        df = df.dropna(axis='index')
        if len(df.index) == 0:
            raise IndexError("Not enough emittance data extracted. Try to give a fill number.")
        dfs[plane] = df

    df = _merge_df_planes(dfs, planes)
    df = _maybe_add_sum_for_planes(df, planes, column_norm_emittance)
    df = _maybe_add_sum_for_planes(df, planes,
                                   lambda p: mean_col(column_norm_emittance(p)),
                                   lambda p: err_col(column_norm_emittance(p)))
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _get_bws_emittances(beam, planes, db, timespan):
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
            time_index = pd.Index(CERNDatetime.from_timestamp(t) for t in x)
            if df is None:
                df = tfs.TfsDataFrame(index=time_index, columns=all_columns, dtype=float)
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


def _get_kick_df(directory, plane):
    def column_action_error(x):
        return err_col(column_action(x))

    try:
        df = _get_new_kick_file(directory, plane)
    except FileNotFoundError:
        LOG.debug("Reading of kickfile failed. Looking for old kickfile.")
        df = _get_old_kick_file(directory, plane)

    df = _maybe_add_sum_for_planes(df, plane, column_action, column_action_error)
    return df[[column_action(plane), column_action_error(plane)]]


def _get_old_kick_file(directory, plane):
    """ Kick files from Beta-Beat.src """
    path = os.path.join(directory, "getkickac.out")
    LOG.debug(f"Reading kickfile '{path}'.'")
    df = tfs.read(path)
    df = df.set_index("TIME")
    df.index = pd.Index(CERNDatetime.from_timestamp(t) for t in df.index)
    rename_dict = {}
    for p in plane:
        rename_dict.update({f"2J{p}RES": column_action(p), f"2J{p}STDRES": err_col(column_action(p))})
    df = df.rename(rename_dict, axis="columns")
    df.loc[:, rename_dict.values()] = df.loc[:, rename_dict.values()] * 1e-6
    return df


def _get_new_kick_file(directory, planes):
    """ Kick files from OMC3 """
    dfs = {p: None for p in planes}
    for plane in planes:
        path = os.path.join(directory, f"{KICKFILE}_{plane.lower()}{TFS_SUFFIX}")
        LOG.debug(f"Reading kickfile '{path}'.'")
        df = tfs.read(path, index=TIME)
        df.index = pd.Index(CERNDatetime.from_cern_utc_string(t) for t in df.index)
        dfs[plane] = df
    return _merge_df_planes(dfs, planes)


def _get_output_dir(directory):
    path = os.path.join(directory, RESULTS_DIR)
    with suppress(IOError):
        os.mkdir(path)
    return path


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


def _add_emittance_to_kicks(plane, energy, kick_df, emittance_df):
    LOG.debug("Retrieving normalized emittance at the kicks.")
    kick_df.headers[HEADER_ENERGY] = energy
    kick_df.headers[HEADER_EMITTANCE_AVERAGE] = ROLLING_AVERAGE_WINDOW
    col_nemittance = column_norm_emittance(plane)
    cols_emitt = [mean_col(col_nemittance), err_col(col_nemittance)]
    cols_kick = [col_nemittance, err_col(col_nemittance)]

    kick_df = kick_df.reindex(columns=kick_df.columns.tolist() + cols_kick)
    idx_emitt = [emittance_df.columns.get_loc(c) for c in cols_emitt]
    for time in kick_df.index:
        idx_kick = emittance_df.index.get_loc(time, method="nearest")
        kick_df.loc[time, cols_kick] = emittance_df.iloc[idx_kick, idx_emitt].values

    # add de-normalized emittance
    normalization = get_proton_gamma(energy) * get_proton_beta(energy)  # norm emittance to emittance
    col_emittance = column_emittance(plane)

    kick_df.headers[header_norm_nominal_emittance(plane)] = LHC_NOMINAL_EMITTANCE
    kick_df.headers[header_nominal_emittance(plane)] = LHC_NOMINAL_EMITTANCE / normalization
    kick_df[col_emittance] = kick_df[col_nemittance] / normalization
    kick_df[err_col(col_emittance)] = kick_df[err_col(col_nemittance)] / normalization
    return kick_df


# Forced DA Fitting ------------------------------------------------------------


def fun_exp_decay(p, x):
    """ p = DA, x[0] = action (2J res), x[1] = emittance"""
    return np.exp(-(p - x[0]/2) / x[1])


def fun_exp_decay_for_curvefit(x, p):
    """ Parameter Swapped. """
    return fun_exp_decay(p, x)


def _fit_exponential(plane, kick_df):
    LOG.debug("Fitting forced da to exponential. ")
    col_action = column_action(plane)
    col_emittance = column_emittance(plane)
    col_losses = rel_col(INTENSITY_LOSSES)

    # get data
    x = [kick_df[col_action], kick_df[col_emittance]]
    y = kick_df[col_losses]

    init_guess = [INITIAL_DA_FIT*kick_df.headers[header_nominal_emittance(plane)]]

    # do prelim fit
    init_fit = _fit_curve(x, y, init_guess)

    # get errors
    err_action = _no_nonzero_errors(kick_df[err_col(col_action)])
    err_emittance = _no_nonzero_errors(kick_df[err_col(col_emittance)])
    err_losses = _no_nonzero_errors(kick_df[err_col(col_losses)])

    sx = [err_action, err_emittance]
    sy = err_losses

    # do odr
    odr = _fit_odr(x, y, sx, sy, init_fit)

    # add DA to kick
    da = odr.beta[0], odr.sd_beta[0]
    kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)] = da
    LOG.info(f"Forced DA in {plane} [m]: {da[0]} ± {da[1]}")

    return kick_df


def _fit_curve(x,y,init):
    """ Preliminary curve fit. """
    fit, cov = scipy.optimize.curve_fit(fun_exp_decay_for_curvefit, x, y, p0=init)
    LOG.info(f"Initial DA fit: {fit} with cov {cov}")
    return fit


def _fit_odr(x,y, sx, sy, init):
    """ ODR Fit """
    # fill zero errors with the minimum error - otherwise fit will not work
    exp_decay_sigma_model = scipy.odr.Model(fun_exp_decay)
    data_model_sigma = scipy.odr.RealData(
        x=x, y=y, sx=sx, sy=sy,
    )
    da_odr = scipy.odr.ODR(data_model_sigma, exp_decay_sigma_model,
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
    """ Converts the DA and the Action into Sigma-Units.
    Hint: Both are currently in units of 2J! """
    LOG.debug("Calculating action and da in sigmas.")
    nominal_emittance = kick_df.headers[header_nominal_emittance(plane)]

    # DA
    da, da_err = kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)]
    da_sigma = np.sqrt(da/nominal_emittance), .5 * da_err / np.sqrt(da * nominal_emittance)
    kick_df.headers[header_da(plane, unit="sigma")], kick_df.headers[header_da_error(plane, unit="sigma")] = da_sigma
    LOG.info(f"Forced DA {plane} in N-sigma: {da_sigma[0]} ± {da_sigma[1]}")

    # Action
    col_action = column_action(plane)
    kick_df[sigma_col(col_action)] = np.sqrt(kick_df[col_action] / nominal_emittance)
    kick_df[err_col(sigma_col(col_action))] = (
            0.5 * kick_df[err_col(col_action)] / np.sqrt(kick_df[col_action] * nominal_emittance)
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
    ax.plot(intensity_df[INTENSITY] / norm,
            marker=".",
            markersize=mpl.rcParams["lines.markersize"] * 0.5,
            fillstyle="full",
            color=colors.get_mpl_color(0),
            label=f'Intensity')

    # plot losses per kick
    normalized_intensity = kick_df.loc[:, [INTENSITY_BEFORE, INTENSITY_AFTER]]/norm
    normalized_intensity_error = kick_df.loc[:, [err_col(INTENSITY_BEFORE), err_col(INTENSITY_AFTER)]]/norm
    normalized_losses = kick_df.loc[:, [INTENSITY_LOSSES, err_col(INTENSITY_LOSSES)]]/norm

    for idx, kick in enumerate(kick_df.index):
        ax.errorbar([kick] * 2,  normalized_intensity.loc[kick, :],
                    yerr=normalized_intensity_error.loc[kick, :],
                    color=colors.get_mpl_color(1),
                    marker='.',
                    linestyle="-",
                    label='__nolegend__' if idx > 0 else "Losses")

        ax.text(kick, .5 * sum(normalized_intensity.loc[kick, :]),
                " -{:.1f}$\pm${:.1f} %".format(*normalized_losses.loc[kick, :]),
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


def _plot_losses(directory, beam, plane, kick_df):
    LOG.debug("Plotting beam losses")
    style.set_style("standard")
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.24, 7.68))

    ax.errorbar(
        kick_df[column_action(plane)] * 1e6,
        kick_df[rel_col(INTENSITY_LOSSES)] * 100,
        xerr=kick_df[err_col(column_action(plane))] * 1e6,
        yerr=kick_df[err_col(rel_col(INTENSITY_LOSSES))] * 100,
        marker=".",
        color=colors.get_mpl_color(0),
        label=f'Kicks')

    ax.set_xlabel(f"$2J_{plane}$ [$\mu$m]")
    ax.set_ylabel(r'Beam Losses [%]')
    annotations.make_top_legend(ax, ncol=3)
    plt.tight_layout()
    annotations.set_name(f"Losses Beam {beam}, Plane {plane}", fig)
    _save_fig(directory, plane, fig, 'losses')


def _plot_emittances(directory, beam, plane, emittance_df, emittance_bws_df, kick_times):
    LOG.debug("Plotting normalized emittances")
    style.set_style("standard")
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.24, 7.68))

    col_norm_emittance = column_norm_emittance(plane)

    bsrt_color = colors.get_mpl_color(0)
    bws_color = colors.get_mpl_color(1)

    ax.plot(emittance_df[col_norm_emittance] * 1e6,  # Actual BSRT measurement
            color=bsrt_color,
            marker='o',
            markeredgewidth=2,
            linestyle='None',
            label=f'From BSRT')

    ax.errorbar(emittance_df.index,
                emittance_df[mean_col(col_norm_emittance)] * 1e6,
                yerr=emittance_df[err_col(col_norm_emittance)],
                color=colors.change_color_brightness(bsrt_color, 0.7),
                marker='',
                label=f'Moving Average (window = {ROLLING_AVERAGE_WINDOW})')

    if len(emittance_bws_df.index):
        for d in BWS_DIRECTIONS:
            label = "__nolegend__" if d == BWS_DIRECTIONS[1] else f"From BWS"
            color = bws_color if d == BWS_DIRECTIONS[1] else colors.change_color_brightness(bws_color, 0.5)
            col_bws_nemittance = column_bws_norm_emittance(plane, d)
            ax.errorbar(
                emittance_bws_df.index,
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


def _plot_da_fit(directory, beam, plane, kick_df):
    """ Plot the Forced Dynamic Aperture fit. """
    LOG.debug("Plotting Dynamic Aperture Fit")
    style.set_style("standard")
    col_action = column_action(plane)
    col_emittance = column_emittance(plane)
    col_intensity = rel_col(INTENSITY_LOSSES)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.24, 7.68))
    ax.errorbar(
        kick_df[col_action] * 1e6,
        kick_df[col_intensity] * 100,
        xerr=kick_df[err_col(col_action)],
        yerr=kick_df[err_col(col_intensity)] * 100,
        marker=".",
        color=colors.get_mpl_color(0),
        label=f'Kicks'
    )

    color = colors.get_mpl_color(1)
    data = kick_df[col_action], kick_df[col_emittance]
    # data = kick_df[col_action], kick_df.headers[header_nominal_emittance(plane)]
    da, da_err = kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)]
    exp_mean = fun_exp_decay(da, data) * 100
    exp_min = fun_exp_decay(da - da_err, data) * 100
    exp_max = fun_exp_decay(da + da_err, data) * 100

    ax.fill_between(data[0]*1e6, exp_min, exp_max,
                    facecolor=mcolors.to_rgba(color, .3))

    da_mu, da_err_mu = significant_digits(da*1e6, da_err*1e6)
    ax.plot(data[0]*1e6, exp_mean, ls="--", c=color,
            label=f'Fit: DA= ${da_mu} \pm {da_err_mu} \mu m$'
            )

    ax.set_xlabel(f"$2J_{{{plane if len(plane) == 1 else ''}}} [\mu m]$")
    ax.set_ylabel(r'Beam Losses [%]')
    ax.set_xlim([0, None])
    ax.set_ylim([0, kick_df[col_intensity].max() * 100 * (1 + YPAD)])
    annotations.make_top_legend(ax, ncol=3)
    plt.tight_layout()
    annotations.set_name(f"DA Fit {beam}, Plane {plane}", fig)
    _save_fig(directory, plane, fig, 'dafit')


def _plot_da_fit_normalized(directory, beam, plane, kick_df):
    """ Plot the Forced Dynamic Aperture fit in normalized units (of sigma). """
    LOG.debug("Plotting Normalized Dynamic Aperture Fit")
    style.set_style("standard")
    col_action = column_action(plane)
    col_emittance = column_emittance(plane)
    col_action_sigma = sigma_col(col_action)
    col_intensity = rel_col(INTENSITY_LOSSES)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.24, 7.68))
    ax.errorbar(
        kick_df[col_action_sigma],
        kick_df[col_intensity] * 100,
        xerr=kick_df[err_col(col_action_sigma)],
        yerr=kick_df[err_col(col_intensity)] * 100,
        marker=".",
        color=colors.get_mpl_color(0),
        label=f'Kicks'
    )

    color = colors.get_mpl_color(1)
    data = kick_df[col_action].sort_values(), np.mean(kick_df[col_emittance])
    da, da_err = kick_df.headers[header_da(plane)], kick_df.headers[header_da_error(plane)]
    da_sigma, da_err_sigma = (kick_df.headers[header_da(plane, unit="sigma")],
                              kick_df.headers[header_da_error(plane, unit="sigma")])

    exp_mean = fun_exp_decay(da, data) * 100
    exp_min = fun_exp_decay(da - da_err, data) * 100
    exp_max = fun_exp_decay(da + da_err, data) * 100

    ax.fill_between(kick_df[col_action_sigma].sort_values(), exp_min, exp_max,
                    facecolor=mcolors.to_rgba(color, .3))

    da_significant = significant_digits(da_sigma, da_err_sigma)
    ax.plot(kick_df[col_action_sigma].sort_values(), exp_mean, ls="--", c=color,
            label=f'Fit: DA= ${da_significant[0]} \pm {da_significant[1]} N_{{\sigma}}$'
            )

    ax.set_xlabel(f"$N_{{\sigma}} = \sqrt{{2J_{{{plane if len(plane) == 1 else ''}}}/\epsilon_{{nominal}}}}$")
    ax.set_ylabel(r'Beam Losses [%]')
    ax.set_xlim([0, None])
    ax.set_ylim([0, kick_df[col_intensity].max() * 100 * (1 + YPAD)])
    annotations.make_top_legend(ax, ncol=3)
    plt.tight_layout()
    annotations.set_name(f"DA Fit Normalized {beam}, Plane {plane}", fig)
    _save_fig(directory, plane, fig, 'dafit_norm')


# Helper ---


def _plot_kicks_and_scale_x(ax, kick_times, pad=20):
    lines.vertical_lines(ax, kick_times, color='grey', linestyle='--', alpha=0.8, marker='', label="Kicks")
    first_kick, last_kick = kick_times.min(), kick_times.max()
    try:
        time_delta = [pd.Timedelta(seconds=pad[i]) for i in range(2)]
    except TypeError:
        time_delta = [pd.Timedelta(seconds=pad) for _ in range(2)]

    ax.set_xlim([first_kick - time_delta[0], last_kick + time_delta[1]])
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
