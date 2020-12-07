"""
SixDesk Utilities
--------------------

Helper Utilities for Autosix.
"""
import subprocess
from contextlib import contextmanager
from pathlib import Path

from omc3.utils import logging_tools

from pylhc.constants.autosix import (SIXDESKLOCKFILE, STAGE_ORDER,
                                     get_workspace_path, get_stagefile_path)
from pylhc.constants.external_paths import SIXDESK_UTILS
from pylhc.htc.mask import find_named_variables_in_mask

LOG = logging_tools.get_logger(__name__)


# Checks  ----------------------------------------------------------------------

def check_mask(mask_text: str, replace_args: dict):
    """ Checks validity/compatibility of the mask and replacement dict. """
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
    if not should_run_stage(jobname, basedir, stage):
        yield False
    else:
        try:
            yield True
        except StageSkip as e:
            if str(e):
                LOG.error(str(e))
        else:
            stage_done(jobname, basedir, stage)


def stage_done(jobname: str, basedir: Path, stage: str):
    """ Append current stage name to stagefile. """
    stage_file = get_stagefile_path(jobname, basedir)
    with open(stage_file, 'a+') as f:
        f.write(f'{stage}\n')


def should_run_stage(jobname: str, basedir: Path, stage: str):
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
    if txt[-1] == STAGE_ORDER[stage_idx - 1]:
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
