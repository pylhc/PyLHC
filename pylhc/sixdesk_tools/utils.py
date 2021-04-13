"""
SixDesk Utilities
--------------------

Helper Utilities for Autosix.
"""
import subprocess
from contextlib import contextmanager
from pathlib import Path

from omc3.utils import logging_tools

from pylhc.constants.autosix import (
    SIXDESKLOCKFILE,
    Stage,
    get_workspace_path,
    get_stagefile_path,
)
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
        raise KeyError(
            "The following keys in the mask were not found for replacement: "
            f"{str(not_in_dict).strip('{}')}"
        )


# Stages -----------------------------------------------------------------------


class StageSkip(Exception):
    pass


@contextmanager
def check_stage(stage: Stage, jobname: str, basedir: Path, max_stage: Stage = None):
    """ Wrapper for stage functions to add stage to stagefile. """
    if not should_run_stage(stage, jobname, basedir, max_stage):
        yield False
    else:
        try:
            yield True
        except StageSkip as e:
            if str(e):
                LOG.error(str(e))
        else:
            stage_done(stage, jobname, basedir)


def stage_done(stage: Stage, jobname: str, basedir: Path):
    """ Append current stage name to stagefile. """
    stage_file = get_stagefile_path(jobname, basedir)
    with open(stage_file, "a+") as f:
        f.write(f"{stage.name}\n")


def should_run_stage(stage: Stage, jobname: str, basedir: Path, max_stage: Stage = None):
    """ Checks if the stage should be run. """
    stage_file = get_stagefile_path(jobname, basedir)
    if not stage_file.exists():
        if stage.value == 0:
            return True
        else:
            LOG.debug(f"Stage {stage} not run because previous stage(s) missing.")
            return False

    with open(stage_file, "r") as f:
        stage_file_txt = f.read().split("\n")
    run_stages = [line.strip() for line in stage_file_txt if line.strip()]

    if stage.name in run_stages:
        LOG.info(f"Stage {stage.name} has already been run. Skipping.")
        return False

    if stage.value == 0:
        return True

    # check if user requested a stop at a certain stage
    if (max_stage is not None) and (stage > max_stage):
        LOG.info(f"Stage {stage.name} would run after requested "
                 f"maximum stage {max_stage.name}. Skipping.")
        return False

    # check if last run stage is also the stage before current stage in stage order
    if run_stages[-1] == Stage(stage-1).name:
        return True

    LOG.debug(f"Stage {stage.name} not run because previous stage(s) missing.")
    return False


# Locks ------------------------------------------------------------------------


def is_locked(jobname: str, basedir: Path, unlock: bool = False):
    """ Checks for sixdesklock-files """
    workspace_path = get_workspace_path(jobname, basedir)
    locks = list(workspace_path.glob(f"**/{SIXDESKLOCKFILE}"))  # list() for repeated usage

    if locks:
        LOG.info("The following folders are locked:")
        for lock in locks:
            LOG.info(f"{str(lock.parent)}")

            with open(lock, "r") as f:
                txt = f.read()
            txt = txt.replace(str(SIXDESK_UTILS), "$SIXUTILS").strip("\n")
            if txt:
                LOG.debug(f" -> locked by: {txt}")

        if unlock:
            for lock in locks:
                LOG.debug(f"Removing lock {str(lock)}")
                lock.unlink()
            return False
        return True
    return False


# Commandline ------------------------------------------------------------------


def start_subprocess(command, cwd=None, ssh: str = None, check_log: str = None):
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
        process = subprocess.Popen(
            ["ssh", ssh, command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
        )

    else:
        # Execute command locally
        LOG.debug(f"Executing command '{' '.join(command)}'")
        process = subprocess.Popen(
            command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
        )

    # Log output
    for line in process.stdout:
        decoded = line.decode("utf-8").strip()
        if decoded:
            LOG.debug(decoded)
            if check_log is not None and check_log in decoded:
                raise OSError(
                    f"'{check_log}' found in last logging message. "
                    "Something went wrong with the last command. Check (debug-)log."
                )

    # Wait for finish and check result
    if process.wait() != 0:
        raise OSError("Something went wrong with the last command. Check (debug-)log.")
