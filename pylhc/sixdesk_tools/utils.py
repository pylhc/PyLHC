import subprocess
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from generic_parser import DotDict
from omc3.utils import logging_tools

from pylhc.htc.mask import find_named_variables_in_mask

LOG = logging_tools.get_logger(__name__)


SIXDESK_UTILS = Path('/afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/')
SETENV_SH = SIXDESK_UTILS / 'set_env.sh'
MAD_TO_SIXTRACK_SH = SIXDESK_UTILS / 'mad6t.sh'
RUNSIX_SH = SIXDESK_UTILS / 'run_six.sh'
RUNSTATUS_SH = SIXDESK_UTILS / 'run_status'

SIXDESKLOCKFILE = 'sixdesklock'

MADX_PATH = Path('/afs/cern.ch/user/m/mad/bin/madx')

SYSENV_MASK = Path(__file__).parent / 'mask_sysenv'
SIXDESKENV_MASK = Path(__file__).parent / 'mask_sixdeskenv'

GIT_REPO = "https://github.com/SixTrack/SixDesk.git"


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
SIXENV_REQUIRED = ['BEAM', 'TURNS', 'AMPMIN', 'AMPMAX', 'AMPSTEP', 'ANGLES']

STAGE_ORDER = ['create_jobs', 'submit_mask', 'check_input', 'submit_sixtrack', 'check_sixtrack_output']
STAGES = DotDict({key: key for key in STAGE_ORDER})

HEADER_BASEDIR = 'BASEDIR'

# Paths ------------------------------------------------------------------------


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


# Checks  ----------------------------------------------------------------------

def check_mask(mask_text: str, replace_args: dict):
    dict_keys = set(replace_args.keys())
    mask_keys = find_named_variables_in_mask(mask_text)
    not_in_dict = mask_keys - dict_keys

    if len(not_in_dict):
        raise KeyError("The following keys in the mask were not found for replacement: "
                       f"{str(not_in_dict).strip('{}')}")


# Stages -----------------------------------------------------------------------

class StageSkip(Exception):
    pass


@contextmanager
def check_stage(stage: str, jobname: str, basedir: Path):
    """ Wrapper for stage functions to add stage to stagefile. """
    if not run_stage(jobname, basedir, stage):
        yield False
    else:
        try:
            yield True
        except StageSkip as e:
            LOG.info(str(e))
        else:
            stage_done(jobname, basedir, stage)


def stage_done(jobname: str, basedir: Path, stage: str):
    stage_file = get_stagefile_path(jobname, basedir)
    with open(stage_file, 'a+') as f:
        f.write(f'{stage}\n')


def run_stage(jobname: str, basedir: Path, stage: str):
    """ Checks if the stage should be run. """
    stage_idx = STAGE_ORDER.index(stage)

    stage_file = get_stagefile_path(jobname, basedir)
    if not stage_file.exists():
        if stage_idx == 0:
            return True
        else:
            LOG.debug(f'Stage {stage} not run because previous stage(s) missing.')
            return False

    with open(stage_file, 'r') as f:
        txt = f.read().split('\n')
    txt = [line.strip() for line in txt if line.strip()]

    if stage in txt:
        LOG.info(f'Stage {stage} has already been run. Skipping.')
        return False

    if stage_idx == 0:
        return True

    # check if last run stage is also the stage before current stage in stage order
    if txt[-1] == STAGE_ORDER[stage_idx-1]:
        return True

    LOG.debug(f'Stage {stage} not run because previous stage(s) missing.')
    return False


# Locks ------------------------------------------------------------------------

def is_locked(jobname: str, basedir: Path, unlock: bool = False):
    """ Checks for sixdesklock-files """
    workspace_path = get_workspace_path(jobname, basedir)
    locks = list(workspace_path.glob(f'**/{SIXDESKLOCKFILE}'))  # list() for repeated usage

    if locks:
        LOG.info('The follwing folders are locked:')
        for lock in locks:
            LOG.info(f"{str(lock.parent)}")

            with open(lock, 'r') as f:
                txt = f.read()
            txt = txt.replace(str(SIXDESK_UTILS), "$SIXUTILS").strip("\n")
            if txt:
                LOG.debug(f' -> locked by: {txt}')

        if unlock:
            for lock in locks:
                LOG.debug(f'Removing lock {str(lock)}')
                lock.unlink()
            return False
        return True
    return False


# Commandline ------------------------------------------------------------------

def start_subprocess(command, cwd=None, ssh: str=None):
    if isinstance(command, str):
        command = [command]

    # convert Paths
    command = [str(c) if isinstance(c, Path) else c for c in command]

    if ssh:
        # Send command to remote machine
        command = " ".join(command)
        if cwd:
            command = f'cd "{cwd}" && {command}'
        LOG.debug(f"Executing command '{command}' on {ssh}")
        process = subprocess.Popen(['ssh', ssh, command], shell=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   cwd=cwd)

    else:
        # Execute command locally
        LOG.debug(f"Executing command '{' '.join(command)}'")
        process = subprocess.Popen(command, shell=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   cwd=cwd)

    # Log output
    for line in process.stdout:
        decoded = line.decode("utf-8").strip()
        if decoded:
            LOG.debug(decoded)

    # Wait for finish and check result
    if process.wait() != 0:
        raise EnvironmentError("Something went wrong with the last command. Check (debug-)log.")
