import os, sys
import pytimber
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from pytimber import pagestore
import datetime
import scipy.odr
import scipy
from contextlib import suppress

from constants.general import PLANES, PLANE_TO_HV, PROTON_MASS, LHC_NORM_EMITTANCE
from constants.forced_da_analysis import *
from utils.time_tools import CERNDatetime
from omc3.omc3.utils import logging_tools
from plotshop import style, lines, annotations, colors
import matplotlib.dates as mdates

LOG = logging_tools.get_logger(__name__)


BWS_DIRECTIONS = ("IN", "OUT")


def main(directory: str, fill: int, beam: int, plane: str, energy: int = 6500, bunch_id: int = None):
    out_dir = get_output_dir(directory)
    gamma = energy / PROTON_MASS  # E = gamma * m0 * c^2
    beta = np.sqrt(1 - (1 / gamma**2))
    emittance = LHC_NORM_EMITTANCE / (beta * gamma)

    kick_df = get_kick_df(directory, plane)
    intensity_df, emittance_df, emittance_bws_df = get_dfs_from_timber(fill, beam, bunch_id, plane)

    _check_all_times_in(kick_df.index, intensity_df.index[0], intensity_df.index[-1])
    _plot_emittances(out_dir, beam, plane, emittance_df, emittance_bws_df, kick_df.index)


def _check_all_times_in(series, start, end):
    if any(s for s in series if s < start or s > end):
        raise ValueError("Some of the kick-times are outside of the fill times! "
                         "Check if correct kick-file or fill number are used.")


# Timber Data ------------------------------------------------------------------


def _get_bctrf_beam_intensity(beam, db, timespan):
    LOG.debug(f"Getting beam intensity from bctfr for beam {beam}.")
    intensity_key = INTENSITY_KEY.format(beam=beam)
    LOG.debug(f"  Key: {intensity_key}")
    x, y = db.get(intensity_key, *timespan)[intensity_key]
    time_index = pd.Index(CERNDatetime.from_timestamp(t) for t in x)
    df = pd.DataFrame(data=y, index=time_index, columns=[INTENSITY])
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _get_bsrt_bunch_emittances(beam, bunch, plane, db, timespan):
    LOG.debug(f"Getting emittance from BSRT for beam {beam}, bunch {bunch} and plane {plane}.")
    bunch_emittance_key = BUNCH_EMITTANCE_KEY.format(beam=beam, plane=PLANE_TO_HV[plane])
    LOG.debug(f"  Key: {bunch_emittance_key}")
    col_norm_emittance = f"{NORM_EMITTANCE}{plane}"
    all_columns = [f"{prefix}{col_norm_emittance}" for prefix in ("", f"{MEAN}", f"{ERR}")]

    x, y = db.get(bunch_emittance_key, *timespan)[bunch_emittance_key]
    time_index = pd.Index(CERNDatetime.from_timestamp(t) for t in x)
    df = pd.DataFrame(index=time_index, columns=all_columns)

    if bunch is None:
        bunch = y.sum(axis=0).argmax()  # first not-all-zero column
        LOG.debug(f"  Found bunch: {bunch}")
    df[col_norm_emittance] = y[:, bunch]

    # remove entries with zero emittance as unphysical
    df = df[df[col_norm_emittance] != 0]
    rolling = df[col_norm_emittance].rolling(window=ROLLING_AVERAGE_WINDOW, center=True)

    df[f'{MEAN}{col_norm_emittance}'] = rolling.mean()
    df[f'{ERR}{col_norm_emittance}'] = rolling.std()
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _get_bws_emittances(beam, plane, db, timespan):
    LOG.debug(f"Getting emittance from BWS for beam {beam} and plane {plane}.")
    all_columns = [f"{NORM_EMITTANCE}{plane}_{direction}" for direction in BWS_DIRECTIONS]
    df = None
    for direction in BWS_DIRECTIONS:
        emittance_key = BWS_EMITTANCE_KEY.format(beam=beam, plane=PLANE_TO_HV[plane], direction=direction)
        LOG.debug(f"  Key: {emittance_key}")
        col_norm_emittance = f"{NORM_EMITTANCE}{plane}_{direction}"

        x, y = db.get(emittance_key, *timespan)[emittance_key]
        time_index = pd.Index(CERNDatetime.from_timestamp(t) for t in x)
        if df is None:
            df = pd.DataFrame(index=time_index, columns=all_columns)

        df[col_norm_emittance] = np.mean(y)
        df[f'{ERR}{col_norm_emittance}'] = np.std(y)
    LOG.debug(f"  Returning dataframe of shape {df.shape}")
    return df


def _plot_emittances(directory, beam, plane, emittance_df, emittance_bws_df, kick_times):
    style.set_style("standard")
    fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=False, figsize=(10, 7))

    col_norm_emittance = f"{NORM_EMITTANCE}{plane}"

    bsrt_color = colors.get_mpl_color(0)
    bws_color = colors.get_mpl_color(1)

    ax.plot(emittance_df[col_norm_emittance],  # Actual BSRT measurement
            color=bsrt_color,
            marker='o',
            markeredgewidth=2,
            linestyle='None',
            label=f'From BSRT')

    ax.errorbar(emittance_df.index,  # Averaged measurement
                emittance_df[f"{MEAN}{col_norm_emittance}"],
                yerr=emittance_df[f"{ERR}{col_norm_emittance}"],
                color=colors.change_color_brightness(bsrt_color, 0.7),
                marker='',
                label=f'Moving Average (window = {ROLLING_AVERAGE_WINDOW})')

    if len(emittance_bws_df.index):
        for d in BWS_DIRECTIONS:
            label = "__nolegend__" if d == BWS_DIRECTIONS[1] else f"From BWS"
            color = bws_color if d == BWS_DIRECTIONS[1] else colors.change_color_brightness(bws_color, 0.5)

            ax.errorbar(emittance_bws_df.index,   # Wire Scanner
                        emittance_bws_df[f"{col_norm_emittance}_{d}"],
                        yerr=emittance_bws_df[f"{ERR}{col_norm_emittance}_{d}"],
                        color=color,
                        label=label,
                        markersize=15,
                        fmt='o'
                        )

    lines.vertical_lines(ax, kick_times, color='grey', linestyle='--', alpha=0.8, marker='', label="Kicks")

    first_kick, last_kick = min(kick_times), max(kick_times)

    time_delta = pd.Timedelta(seconds=20)
    ax.set_xlim([first_kick - time_delta, last_kick + time_delta])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\epsilon_{n}$ $[\mu m]$')
    # annotations.figure_title(f"Emittance Beam {beam}, {plane}-Plane", ax, position="bottom", pad=0.015, fontweight="bold")
    annotations.make_top_legend(ax, ncol=2)
    annotations.set_annotation(f"Date: {first_kick.strftime('%Y-%m-%d')}", ax, "left")
    plt.tight_layout()
    try:
        fig.savefig(os.path.join(directory, f"emittance_b{beam}_{plane.lower()}.pdf"))
        fig.savefig(os.path.join(directory, f"emittance_b{beam}_{plane.lower()}.png"))
    except IOError:
        LOG.error("Couldn't create output files for emittance plots!")
    plt.show()


def get_dfs_from_timber(fill: int, beam: int, bunch: int, plane: str):
    # open Logging DB for specific fill and db for storing data
    db = pytimber.LoggingDB()
    filldata = db.getLHCFillData(fill)
    timespan = filldata['startTime'], filldata['endTime']
    intensity_df = _get_bctrf_beam_intensity(beam, db, timespan)
    emittance_bws_df = _get_bws_emittances(beam, plane, db, timespan)
    emittance_df = _get_bsrt_bunch_emittances(beam, bunch, plane,  db, timespan)
    return intensity_df, emittance_df, emittance_bws_df


# Kick Data --------------------------------------------------------------------


def get_kick_df(directory, plane):
    try:
        df = _get_new_kick_file(directory, plane)
    except FileNotFoundError:
        LOG.debug("Reading of kickfile failed. Looking for old kickfile.")
        df = _get_old_kick_file(directory, plane)
    return df[[f"2J{plane}RES", f"ERR2J{plane}RES"]]  # TODO: get from omc3


def _get_old_kick_file(directory, plane):
    path = os.path.join(directory, "getkickac.out")
    LOG.debug(f"Reading kickfile '{path}'.'")
    df = tfs.read(path)
    df = df.rename({f"2J{plane}STDRES": f"ERR2J{plane}RES"}, axis="columns")
    df = df.set_index("TIME")
    df.index = pd.Index(CERNDatetime.from_timestamp(t) for t in df.index)
    return df


def _get_new_kick_file(directory, plane):
    path = os.path.join(directory, f"{KICKFILE}_{plane.lower()}{TFS_SUFFIX}")
    LOG.debug(f"Reading kickfile '{path}'.'")
    df = tfs.read(path, index=TIME)
    df.index = pd.Index(CERNDatetime.from_cern_utc_string(t) for t in df.index)
    return df


def get_output_dir(directory):
    path = os.path.join(directory, RESULTS_DIR)
    with suppress(IOError):
        os.mkdir(path)
    return path


if __name__ == '__main__':
    FILL_NUMBER = 7391
    BEAM = 1
    BUNCH_ID = 0
    ANALYSISDIR = '/media/jdilly/Storage/Repositories/Gui_Output/2019-08-09/LHCB1/Results/b1_amplitude_det_vertical_all'
    energy = 6500  # GeV
    PLANE = PLANES[0]
    main(directory=ANALYSISDIR, fill=FILL_NUMBER, beam=BEAM, plane=PLANE)

exit()


### get intensity at kicktime for vertical kicks
waittime = 3

fbct_v_losses = np.zeros((len(kicks_df['TIME']),2))


for i, time in enumerate(kicks_df['TIME']):

    before_kick = fbct_df['INTENSITY'].iloc[ fbct_df.index.get_loc( float(time - 30), method='ffill'):fbct_df.index.get_loc( float(time - 5), method='ffill')  ]
    after_kick = fbct_df['INTENSITY'].iloc[ fbct_df.index.get_loc( float(time + waittime), method='ffill') : fbct_df.index.get_loc( float(time + waittime + 15), method='ffill') ]

    before_kick_av = np.mean( before_kick )
    before_kick_std = np.std( before_kick )
    after_kick_av = np.mean( after_kick )
    after_kick_std = np.std( after_kick )


    losses = before_kick_av - after_kick_av
    losses_std = np.sqrt(before_kick_std**2 + after_kick_std**2)

    fbct_v_losses[i,:] = losses  / before_kick_av , np.sqrt( (losses  / before_kick_av)**2 * ( ( losses_std / losses )**2 + ( before_kick_std / before_kick_av )**2  ) )


kicks_df = kicks_df.assign(Losses=fbct_v_losses[:,0])
kicks_df = kicks_df.assign(Losses_std=fbct_v_losses[:,1])

kicks_df['Losses_std'] = kicks_df['Losses_std'].replace(to_replace=0, method='bfill' )

kicks_df['Losses_std'] = kicks_df['Losses_std'] + 0.01

kicks_df = kicks_df.assign( cumulative_Losses=kicks_df['Losses'].cumsum())
kicks_df = kicks_df.assign( cumulative_Losses_std=kicks_df['Losses_std'])

kicks_df = kicks_df.assign( adding_Losses=kicks_df['Losses'].shift(1, fill_value=0) )
kicks_df['adding_Losses'] = np.where( kicks_df['adding_Losses'] < 0.0001, 0, kicks_df['adding_Losses'] ) + kicks_df['Losses']

kicks_df = kicks_df.assign( adding_Losses_std=kicks_df['Losses_std'])

print( kicks_df.tail(5) )

### get emittance at kicktime for vertical kicks
waittime = 3

kick_emittance = np.zeros((len(kicks_df['TIME']),2))


for i, time in enumerate(kicks_df['TIME']):

    kick_emittance[i,:] = bsrt_v_df['NEMITTANCE_V_AV7'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill')  ], bsrt_v_df['NEMITTANCE_V_STD7'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill') ]

#     kick_emittance[i,:] = bsrt_v_sigma_df['NEMITTANCE_V'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill')  ], bsrt_v_sigma_df['NEMITTANCE_V_STD'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill') ]



kicks_df = kicks_df.assign(nemittance=kick_emittance[:,0])
kicks_df = kicks_df.assign(nemittance_std=kick_emittance[:,1])

print( kicks_df.tail(5) )

fig, ax = plt.subplots( ncols=1, nrows=1, sharex=False, sharey=False, figsize=(9,9) )


ax.errorbar( kicks_df['2JYRES'], kicks_df['Losses']*100, xerr=kicks_df['2JYSTDRES'], yerr=kicks_df['Losses_std']*100, label='Single losses' )


ax.errorbar( kicks_df['2JYRES'], kicks_df['cumulative_Losses']*100, xerr=kicks_df['2JYSTDRES'], yerr=kicks_df['cumulative_Losses_std']*100, label='Cumulative losses' )


ax.errorbar( kicks_df['2JYRES'], kicks_df['adding_Losses']*100, xerr=kicks_df['2JYSTDRES'], yerr=kicks_df['adding_Losses_std']*100, label='added Losses from prev. Kick' )


ax.set_xlabel(r'$2J_y$')
ax.set_ylabel(r'Losses in %')
ax.legend()

plt.tight_layout()
plt.show()


#%%
kicks_df['Kick_sigma']  = ( np.sqrt( kicks_df['2JYRES']*10**-6 /kicks_df['nemittance']  ) )
kicks_df['Kick_sigma_std']  = 0.5 * kicks_df['2JYSTDRES']*10**-6 / np.sqrt( kicks_df['2JYRES']*10**-6 * kicks_df['nemittance'] )

kicks_df['Kick_sigma_cumulative']  = ( np.sqrt( kicks_df['2JYRES']*10**-6 /kicks_df['nemittance'].iloc[0]  ) )
kicks_df['Kick_sigma_cumulative_std']  = 0.5 * kicks_df['2JYSTDRES']*10**-6 / np.sqrt( kicks_df['2JYRES']*10**-6 * kicks_df['nemittance'].iloc[0] )

kicks_df['Kick_sigma_nominal']  = (kicks_df['Kick_sigma'] * np.sqrt(  kicks_df['nemittance']/emittance  ) )
kicks_df['Kick_sigma_nominal_std']  = 0.5 * kicks_df['2JYSTDRES']*10**-6 / np.sqrt( kicks_df['2JYRES']*10**-6 * emittance )

print( kicks_df['Kick_sigma'] )

print( kicks_df.tail(5) )

# losses formula assuming kicks and DA are normalised to sigma
def exp_decay_sigma(p, x):

    return np.exp( (x - p)/(2*1) )
# fit forced DA for single losses
kicks_df = kicks_df.iloc[0:11, :]
exp_decay_sigma_model = scipy.odr.Model(exp_decay_sigma)
data_model_sigma = scipy.odr.RealData( x = kicks_df['Kick_sigma'] ,
                                y = kicks_df['Losses'],
                                sx = kicks_df['Kick_sigma_std'],
                                sy = kicks_df['Losses_std'] )

da_odr = scipy.odr.ODR( data_model_sigma, exp_decay_sigma_model, beta0=[4.] )
# da_odr.set_job( fit_type=2 )
odr_output = da_odr.run()
odr_output.pprint()


#%%
# plot losses and forced DA fit

DA = odr_output.beta[0]
DA_err =  odr_output.sd_beta[0]

print('-------------------------------------------------------------')
print('DA vertical: ' + str(DA) + ' +- ' + str(DA_err))
print('-------------------------------------------------------------')

# plot losses and forced DA fit

DA = odr_output.beta[0]
DA_err =  odr_output.sd_beta[0]

print('-------------------------------------------------------------')
print('DA vertical: ' + str(DA) + ' +- ' + str(DA_err))
print('-------------------------------------------------------------')

# plot Kicks over losses
fig, ax = plt.subplots( ncols=1, nrows=1, sharex=False, sharey=False, figsize=(10,10) )


ax.errorbar( kicks_df['Kick_sigma'], kicks_df['Losses']*100, xerr=kicks_df['Kick_sigma_std'], yerr=kicks_df['Losses_std']*100,
            label='AC dipole', capsize=10, linestyle='', color='darkgreen', elinewidth=3., capthick=4  )

fitish = np.zeros(len(kicks_df["Kick_sigma"]))

for i in range(len(fitish)):
    fitish[i] = exp_decay_sigma( odr_output.beta[0], kicks_df['Kick_sigma'].iloc[i]  )

ax.plot( kicks_df['Kick_sigma'], fitish*100, label='Fit: DA= ${:.1f} \pm {:.1f} \sigma_{{measured}}$'.format( DA, DA_err ), color='lawngreen', linewidth=4 )

ax.set_xlabel(r'$ N \sigma_{y, measured}$', fontsize=25)
ax.set_ylabel(r'Losses in % ', fontsize=25)

ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(loc='upper left', fontsize=25)
# ax.set_ylim([0,20])
# ax.set_xlim([0,12])
# ax_doub.set_xlim([0,6])
plt.tight_layout()
plt.savefig( 'ac_dipole_losses_sigma.eps' )
plt.show()

exit()


exp_decay_sigma_model = scipy.odr.Model(exp_decay_sigma)
data_model_sigma = scipy.odr.RealData( x = kicks_df['Kick_sigma_cumulative'] ,
                                y = kicks_df['cumulative_Losses'],
                                sx = kicks_df['Kick_sigma_cumulative_std'],
                                sy = kicks_df['cumulative_Losses_std'] )

da_odr = scipy.odr.ODR( data_model_sigma, exp_decay_sigma_model, beta0=[4.] )
# da_odr.set_job( fit_type=2 )
odr_output = da_odr.run()
odr_output.pprint()

