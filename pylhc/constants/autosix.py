"""
Constants: Autosix
----------------------------------

Collections of constants and paths used in autosix.

:module: constants.autosix
:author: jdilly

"""
from pathlib import Path

from generic_parser import DotDict

from pylhc.constants.external_paths import SIXDESK_UTILS, MADX_BIN

# Program Paths ----------------------------------------------------------------

SETENV_SH = SIXDESK_UTILS / 'set_env.sh'
MAD_TO_SIXTRACK_SH = SIXDESK_UTILS / 'mad6t.sh'
RUNSIX_SH = SIXDESK_UTILS / 'run_six.sh'
RUNSTATUS_SH = SIXDESK_UTILS / 'run_status'
DOT_PROFILE = SIXDESK_UTILS / 'dot_profile'
SIXDB = SIXDESK_UTILS.parent / 'externals' / 'SixDeskDB' / 'sixdb'
SIXDESKLOCKFILE = 'sixdesklock'
SYSENV_MASK = Path(__file__).parent / 'mask_sysenv'
SIXDESKENV_MASK = Path(__file__).parent / 'mask_sixdeskenv'


# Constants and Requirements ---------------------------------------------------

HEADER_BASEDIR = 'BASEDIR'

DEFAULTS = dict(
    python='python3',
    da_turnstep=100,
    executable=MADX_BIN,
)

# Sixenv ---
SIXENV_REQUIRED = ['BEAM', 'TURNS', 'AMPMIN', 'AMPMAX', 'AMPSTEP', 'ANGLES']
SIXENV_DEFAULT = dict(
    RESUBMISSION=0,  # 0: never, 1: if fort.10, 2: always
    PLATFORM='HTCondor',
    LOGLEVEL=0,  # 0: basic + errors , 1: + info, >=2: + debug
    FIRSTSEED=1,
    LASTSEED=60,
    ENERGY='col',  # 'col' or 'inj'
    NPAIRS=30,  # 1-32 particle pairs
    EMITTANCE=3.75,  # normalized emittance
    DIMENSIONS=6,  # Phase-Space dimensions
    WRITEBINS=500,
)

# Stages ---
STAGE_ORDER = ['create_jobs', 'submit_mask', 'check_input',
               'submit_sixtrack', 'check_sixtrack_output',
               'sixdb_load', 'sixdb_cmd', 'final']
STAGES = DotDict({key: key for key in STAGE_ORDER})


# Workspace Paths --------------------------------------------------------------

def get_workspace_path(jobname: str, basedir: Path) -> Path:
    return basedir / f'workspace-{jobname}'


def get_scratch_path(basedir: Path) -> Path:
    return basedir / f'scratch-0'


def get_sixjobs_path(jobname: str, basedir: Path) -> Path:
    return get_workspace_path(jobname, basedir) / 'sixjobs'


def get_masks_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / 'mask'


def get_stagefile_path(jobname: str, basedir: Path):
    return get_sixjobs_path(jobname, basedir) / 'stages_completed.txt'


def get_sixtrack_input_path(jobname: str, basedir: Path):
    return get_sixjobs_path(jobname, basedir) / 'sixtrack_input'


def get_mad6t_mask_path(jobname: str, basedir: Path):
    return get_sixtrack_input_path(jobname, basedir) / 'mad6t.sh'


def get_mad6t1_mask_path(jobname: str, basedir: Path):
    return get_sixtrack_input_path(jobname, basedir) / 'mad6t1.sh'