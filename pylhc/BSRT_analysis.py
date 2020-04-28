'''
BSRT analysis
---------------------------



'''
import glob
from pathlib import Path
import gzip
import pickle
import datetime
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import parse
import tfs
from generic_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from pylhc.constants.general import PLANES, FILE_SUFFIX, TIME_COLUMN

LOG = logging_tools.get_logger(__name__)


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
        exit()

    _plot_fit_variables(opt, bsrt_df)
    _plot_full_crosssection(opt, bsrt_df)
    _plot_crosssection_for_timesteps(opt, bsrt_df)
    _plot_auxilliary_variables(opt, bsrt_df)

# File Name Functions ----------------------------------------------------------

def _get_bsrt_logger_fname(beam, timestamp):
    return f'data_BSRT_{beam}_{timestamp}.dat.gz'


def _get_bsrt_tfs_fname(beam):
    return f'data_BSRT_{beam}.tfs'

# File Handling  ---------------------------------------------------------------

def _select_files(opt):
    files_df = pd.DataFrame(data={'FILES':glob.glob(Path(opt.directory)/_get_bsrt_logger_fname(opt.beam, '*'))})

    files_df = files_df.assign(TIMESTAMP=[_get_timestamp_from_filename(opt, f) for f in files_df['FILES']])
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


def _get_timestamp_from_filename(opt, filename):

    fname = Path(filename).name
    format_string = _get_bsrt_logger_fname(opt.beam, '{}-{}-{}-{}-{}-{}.{}')
    year, month, day, hour, minute, second, microsecond = map(int, parse.parse(format_string, fname))
    return datetime.datetime(year, month, day, hour, minute, second, microsecond, tzinfo=pytz.timezone('UTC'))


def _load_pickled_data(opt, files_df):
    merged_df = pd.DataFrame()
    for bsrtfile in files_df['FILES']:
        data = pickle.load(gzip.open(bsrtfile, 'rb'))
        for entry in data:
            merged_df = merged_df.append(entry, ignore_index=True)

    if opt.outputdir is not None:
        tfs.write(Path(opt.outputdir)/_get_bsrt_tfs_fname(opt.beam), merged_df)

    return merged_df

# Plotting Functions  ----------------------------------------------------------

def _plot_fit_variables(opt, bsrt_df):
    pass


def _plot_full_crosssection(opt, bsrt_df):
    pass


def _plot_crosssection_for_timesteps(opt, bsrt_df):
    pass


def _plot_auxilliary_variables(opt, bsrt_df):
    pass

# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    main()
