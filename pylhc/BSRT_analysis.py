'''
BSRT analysis
---------------------------

Processes the output files of BSRT_logger.py for a given timeframe, 
returns them in as a tfs for further processing.
Additionally, plots for quick checks of fit parameters, auxiliary variables and 
beam evolution are generated.
Provided a tfs file with timestamps, plots of the 2D distribution and comparision 
of fit parameter to crosssections are added.
Plots functions return figures and can be imported in e.g. IPython notebooks for plot tweaking.
'''
import sys
import glob
from pathlib import Path
import gzip
import pickle
import datetime
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parse
import tfs
from generic_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from pylhc.constants.general import FILE_SUFFIX, TIME_COLUMN

LOG = logging_tools.get_logger(__name__)
PLOT_FILE_SUFFIX = '.pdf'
TIME_FORMAT = '%Y/%m/%d %H:%M:%S.%f'

def get_params():
    return EntryPointParameters(
        directory=dict(
            flags=['-d', "--directory"],
            required=True,
            type=str,
            help="Directory containing the logged BSRT files."
        ),
        beam=dict(
            flags=['-b', "--beam"],
            required=True,
            choices=['B1', 'B2'],
            type=str,
            help="Beam for which analysis is performed."
        ),
        outputdir=dict(
            flags=['-o', "--outputdir"],
            type=str,
            help="Directory in which plots and dataframe will be saved in. If omitted, no data will be saved."
        ),
        starttime=dict(
            flags=["--starttime"],
            type=int,
            help="Start of time window for analysis in milliseconds UTC."
        ),
        endtime=dict(
            flags=["--endtime"],
            type=int,
            help="End of time window for analysis in milliseconds UTC."
        ),
        kick_df=dict(
            flags=["--kick_df"],
            type=str,
            help=f"TFS with column {TIME_COLUMN} with time stamps to be added in the plots. Additionally, crossection at these timestamps will be plotted."
        ),
        show_plots=dict(
            flags=["--show_plots"],
            type=bool,
            help="Show BSRT plots."
        ),
    )



@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting BSRT analysis.")

    files_df = _select_files(opt)
    bsrt_df = _load_pickled_data(opt, files_df)

    if not opt.show_plots or opt.outputdir is  None:
        LOG.info("Neither plot display nor outputdir was selcted. Plotting is omitted")
        sys.exit()

    plot_fit_variables(opt, bsrt_df)
    plot_full_crosssection(opt, bsrt_df)
    plot_auxiliary_variables(opt, bsrt_df)
    if opt.kick_df is not None:
        plot_crosssection_for_timesteps(opt, bsrt_df)

# File Name Functions ----------------------------------------------------------

def _get_bsrt_logger_fname(beam, timestamp):
    return f'data_BSRT_{beam}_{timestamp}.dat.gz'


def _get_bsrt_tfs_fname(beam):
    return f'data_BSRT_{beam}{FILE_SUFFIX}'


def _get_fitvar_plot_fname(beam):
    return f'plot_BSRT_FitVariables_{beam}{PLOT_FILE_SUFFIX}'


def _get_2dcrossection_plot_fname(beam):
    return f'plot_BSRT_2DCrossection_{beam}{PLOT_FILE_SUFFIX}'


def _get_crossection_plot_fname(beam, timestamp):
    return f'plot_BSRT_Crossection_{timestamp}_{beam}{PLOT_FILE_SUFFIX}'


def _get_auxiliary_var_plot_fname(beam):
    return f'plot_BSRT_auxVariables_{beam}{PLOT_FILE_SUFFIX}'

# File Handling  ---------------------------------------------------------------

def _select_files(opt):
    files_df = pd.DataFrame(data={'FILES':glob.glob(Path(opt.directory)/_get_bsrt_logger_fname(opt.beam, '*'))})

    files_df = files_df.assign(TIMESTAMP=[
        _get_timestamp_from_name(Path(f).name, _get_bsrt_logger_fname(opt.beam, '{}-{}-{}-{}-{}-{}.{}'))
        for f in files_df['FILES']
        ]
        )
    files_df = files_df.assign(TIME=[f.timestamp() for f in files_df['TIMESTAMP']])

    files_df = files_df.sort_values(by=['TIME']).reset_index(drop=True).set_index('TIME')

    if opt.endtime is not None and opt.starttime is not None:
        assert opt.endtime >= opt.starttime

    indices = []
    for time, fct in zip([opt.startime, opt.endtime], ['first_valid_index', 'last_valid_index']):
        indices.append(
            _get_closest_index(files_df, time if time is not None else getattr(files_df, fct)())
        )

    return files_df.iloc[indices[0]:indices[1]]


def _get_closest_index(df, time):
    return  df.index.get_loc(time, method='nearest')


def _get_timestamp_from_name(name, formatstring):
    year, month, day, hour, minute, second, microsecond = map(int, parse.parse(formatstring, name))
    return datetime.datetime(year, month, day, hour, minute, second, microsecond, tzinfo=pytz.timezone('UTC'))


def _load_pickled_data(opt, files_df):
    merged_df = pd.DataFrame()
    for bsrtfile in files_df['FILES']:
        data = pickle.load(gzip.open(bsrtfile, 'rb'))
        for entry in data:
            merged_df = merged_df.append(entry, ignore_index=True)

    merged_df = merged_df.set_index(pd.to_datetime(merged_df['acqTime'], format=TIME_FORMAT))
    if opt.outputdir is not None:
        tfs.write(Path(opt.outputdir)/_get_bsrt_tfs_fname(opt.beam), merged_df)

    return merged_df

# Plotting Functions  ----------------------------------------------------------

def plot_fit_variables(opt, bsrt_df):

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 9), sharex=True)

    ax[0, 0].plot(bsrt_df.index, [entry[2] for entry in bsrt_df['lastFitResults']])
    ax[0, 0].set_title('Horizontal Amplitude')
    ax[0, 0].set_ylim(bottom=0)

    ax[0, 1].plot(bsrt_df.index, [entry[3] for entry in bsrt_df['lastFitResults']])
    ax[0, 1].set_title('Horizontal Center')

    ax[0, 2].plot(bsrt_df.index, [entry[4] for entry in bsrt_df['lastFitResults']])
    ax[0, 2].set_title('Horizontal Sigma')
    ax[0, 2].set_ylim(bottom=0)

    ax[1, 0].plot(bsrt_df.index, [entry[7] for entry in bsrt_df['lastFitResults']])
    ax[1, 0].set_title('Vertical Amplitude')
    ax[1, 0].set_ylim(bottom=0)

    ax[1, 1].plot(bsrt_df.index, [entry[8] for entry in bsrt_df['lastFitResults']])
    ax[1, 1].set_title('Vertical Center')

    ax[1, 2].plot(bsrt_df.index, [entry[9] for entry in bsrt_df['lastFitResults']])
    ax[1, 2].set_title('Vertical Sigma')
    ax[1, 2].set_ylim(bottom=0)

    plt.tight_layout()

    if opt.outputdir is not None:
        plt.savefig(Path(opt.outputdir)/_get_fitvar_plot_fname(opt.beam))
    if opt.show_plots:
        plt.show()
    return fig


def flattend_column(df, col):
    flat_column = []
    for _, entry in df.iterrows():
        flat_column = [*flat_column, *entry[col]]
    return flat_column


def add_xcols(df):
    x1_col=[]
    x2_col=[]

    for idx, row in df.iterrows():
        x1_col.append((np.ones(len(row['projPositionSet1']))*idx))
        x2_col.append((np.ones(len(row['projPositionSet2']))*idx))

    df['XCol_Set1'] = x1_col
    df['XCol_Set2'] = x2_col
    return df


def plot_full_crosssection(opt, bsrt_df):

    bsrt_df = add_xcols(bsrt_df)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))

    ax[0].scatter(x=flattend_column(bsrt_df, 'XCol_Set1'),
                  y=flattend_column(bsrt_df, 'projPositionSet1'),
                  s=np.ones(len(flattend_column(bsrt_df, 'projDataSet1')))*0.5,
                  c=flattend_column(bsrt_df, 'projDataSet1'),
                  marker='s',
                  cmap='inferno')
    ax[0].plot(bsrt_df.index, [entry[3] for entry in bsrt_df['lastFitResults']],
               color='white', linewidth=0.5)
    ax[0].plot(bsrt_df.index, [entry[3]+entry[4] for entry in bsrt_df['lastFitResults']],
               color='white', linestyle='--', linewidth=0.3)
    ax[0].plot(bsrt_df.index, [entry[3]-entry[4] for entry in bsrt_df['lastFitResults']],
               color='white', linestyle='--', linewidth=0.3)
    ax[0].set_title('Horizontal Crossection')

    ax[1].scatter(x=flattend_column(bsrt_df, 'XCol_Set2'),
                  y=flattend_column(bsrt_df, 'projPositionSet2'),
                  s=np.ones(len(flattend_column(bsrt_df, 'projDataSet2')))*0.5,
                  c=flattend_column(bsrt_df, 'projDataSet2'),
                  marker='s',
                  cmap='inferno')
    ax[1].plot(bsrt_df.index, [entry[8] for entry in bsrt_df['lastFitResults']],
               color='white', linewidth=0.5)
    ax[1].plot(bsrt_df.index, [entry[8]+entry[9] for entry in bsrt_df['lastFitResults']],
               color='white', linestyle='--', linewidth=0.3)
    ax[1].plot(bsrt_df.index, [entry[8]-entry[9] for entry in bsrt_df['lastFitResults']],
               color='white', linestyle='--', linewidth=0.3)
    ax[1].set_title('Vertical Crossection')

    plt.tight_layout()
    
    if opt.outputdir is not None:
        plt.savefig(Path(opt.outputdir)/_get_2dcrossection_plot_fname(opt.beam))
    if opt.show_plots:
        plt.show()
    return fig


def _gauss(x, *p):
    a, b, c = p
    y = a*np.exp(-(x - b)**2/(2. * c**2.))

    return y


def _reshaped_imageset(df):
    return np.reshape(df['imageSet'], (df['acquiredImageRectangle'][3], df['acquiredImageRectangle'][2]))


def plot_crosssection_for_timesteps(opt, bsrt_df):
    kick_df = tfs.read(opt.kick_df)
    figlist=[]

    if TIME_COLUMN not in bsrt_df.columns:
        raise AssertionError(f'Column {TIME_COLUMN} not found in kick_tfs.')

    for timestamp in pd.to_datetime(kick_df[TIME_COLUMN], format=TIME_FORMAT):

        data_row = bsrt_df.iloc[_get_closest_index(bsrt_df, timestamp)]

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,18))
        
        ax[0].image(_reshaped_imageset(data_row), cmap='hot', interpolation='nearest')
        ax[0].set_title(f'2D Pixel count, Timestamp: {timestamp}')
        
        ax[1].plot(data_row['projPositionSet1'], data_row['projDataSet1'], color='darkred')
        ax[1].plot(data_row['projPositionSet1'],
                     _gauss(data_row['projPositionSet1'],
                            data_row['lastFitResults'][2],
                            data_row['lastFitResults'][3],
                            data_row['lastFitResults'][4]),
                     color='darkgreen',
                     label='Gaussian Fit')
        ax[1].set_ylim(bottom=0)
        ax[1].legend()
        ax[1].set_title('Horizontal Projection')

        ax[2].plot(data_row['projPositionSet2'], data_row['projDataSet2'], color='darkred')
        ax[2].plot(data_row['projPositionSet2'],
                     _gauss(data_row['projPositionSet2'],
                            data_row['lastFitResults'][7],
                            data_row['lastFitResults'][8],
                            data_row['lastFitResults'][9]),
                      color='darkgreen',
                      label='Gaussian Fit')
        ax[2].set_ylim(bottom=0)
        ax[2].legend()
        ax[2].set_title('Vertical Projection')

        plt.tight_layout()
        if opt.outputdir is not None:
            plt.savefig(Path(opt.outputdir)/_get_crossection_plot_fname(opt.beam, timestamp))
        if opt.show_plots:
            plt.show()
        figlist.append(fig)
    return figlist



def plot_auxiliary_variables(opt, bsrt_df):

    fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(9, 20), sharex=True)
 
    ax[0].plot(bsrt_df.index, bsrt_df['acqCounter'])
    ax[0].set_title('acqCounter')

    ax[1].plot(bsrt_df.index, bsrt_df['lastAcquiredBunch'])
    ax[1].set_title('lastAcquiredBunch')

    ax[2].plot(bsrt_df.index, bsrt_df['opticsResolutionSet1'], color='red', label='opticsResolutionSet1')
    ax[2].twinx().plot(bsrt_df.index, bsrt_df['opticsResolutionSet2'], color='blue', label='opticsResolutionSet2')
    ax[2].legend()
    ax[2].set_title('opticsResolution')

    ax[3].plot(bsrt_df.index, bsrt_df['cameraGainVoltage'])
    ax[3].set_title('cameraGainVoltage')

    ax[4].plot(bsrt_df.index, bsrt_df['imageCenterSet1'], color='red', label='imageCenterSet1')
    ax[4].twinx().plot(bsrt_df.index, bsrt_df['imageCenterSet2'], color='blue', label='imageCenterSet2')
    ax[4].legend()
    ax[4].set_title('imageCenter')

    ax[5].plot(bsrt_df.index, bsrt_df['betaTwissSet1'], color='red', label='betaTwissSet1')
    ax[5].twinx().plot(bsrt_df.index, bsrt_df['betaTwissSet2'], color='blue', label='betaTwissSet2')
    ax[5].legend()
    ax[5].set_title('betaTwissSet')

    ax[6].plot(bsrt_df.index, bsrt_df['imageScaleSet1'], color='red', label='imageScaleSet1')
    ax[6].twinx().plot(bsrt_df.index, bsrt_df['imageScaleSet2'], color='blue', label='imageScaleSet2')
    ax[6].legend()
    ax[6].set_title('imageScale')

    if opt.outputdir is not None:
        plt.savefig(Path(opt.outputdir)/_get_auxiliary_var_plot_fname(opt.beam))
    if opt.show_plots:
        plt.show()
    return fig

# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    main()
