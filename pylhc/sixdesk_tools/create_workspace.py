"""
Create SixDesk Workspace
-----------------------------------

Tools to setup the workspace for sixdesk.
"""
import shutil
from pathlib import Path

import numpy as np
from omc3.utils import logging_tools

from pylhc.constants.autosix import (
    SETENV_SH,
    SIXENV_DEFAULT,
    SIXENV_REQUIRED,
    SEED_KEYS,
    get_workspace_path,
    get_scratch_path,
    get_sixjobs_path,
    get_masks_path,
    get_mad6t_mask_path,
    get_mad6t1_mask_path,
    get_autosix_results_path,
    get_sysenv_path,
    get_sixdeskenv_path,
)
from pylhc.sixdesk_tools.utils import start_subprocess

SYSENV_MASK = Path(__file__).parent / "mask_sysenv"
SIXDESKENV_MASK = Path(__file__).parent / "mask_sixdeskenv"

LOG = logging_tools.get_logger(__name__)


# Main -------------------------------------------------------------------------


def create_job(jobname: str, basedir: Path, mask_text: str, binary_path: Path, ssh: str = None, **kwargs):
    """ Create environment and individual jobs/masks for SixDesk to send to HTC. """
    _create_workspace(jobname, basedir, ssh=ssh)
    _create_sysenv(jobname, basedir, binary_path=binary_path)
    _create_sixdeskenv(jobname, basedir, **kwargs)
    _write_mask(jobname, basedir, mask_text, **kwargs)
    LOG.info("Workspace prepared.")


def init_workspace(jobname: str, basedir: Path, ssh: str = None):
    """ Initializes the workspace with sixdeskenv and sysenv. """
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    start_subprocess([SETENV_SH, "-s"], cwd=sixjobs_path, ssh=ssh)
    LOG.info("Workspace initialized.")


def remove_twiss_fail_check(jobname: str, basedir: Path):
    """ Comments out the "Twiss fail" check from mad6t.sh """
    LOG.info("Applying twiss-fail hack.")
    for mad6t_path in (
        get_mad6t_mask_path(jobname, basedir),
        get_mad6t1_mask_path(jobname, basedir),
    ):
        with open(mad6t_path, "r") as f:
            lines = f.readlines()

        check_started = False
        for idx, line in enumerate(lines):
            if line.startswith('grep -i "TWISS fail"'):
                check_started = True

            if check_started:
                lines[idx] = f"# {line}"
                if line.startswith("fi"):
                    break
        else:
            LOG.info(f"'TWISS fail' not found in {mad6t_path.name}")
            continue

        with open(mad6t_path, "w") as f:
            f.writelines(lines)


# Helper -----------------------------------------------------------------------


def _create_workspace(jobname: str, basedir: Path, ssh: str = None):
    """ Create workspace structure (with default files). """
    workspace_path = get_workspace_path(jobname, basedir)
    scratch_path = get_scratch_path(basedir)
    LOG.info(f'Creating new workspace in "{str(workspace_path)}"')

    if workspace_path.exists():
        LOG.warning(f'Workspace in "{str(workspace_path)}" already exists. ')
        LOG.info("Do you want to delete the old workspace? [y/N]")
        user_answer = input()
        if user_answer.lower().startswith("y"):
            shutil.rmtree(workspace_path)
            try:
                shutil.rmtree(scratch_path)
            except FileNotFoundError:
                pass
        else:
            LOG.warning("Keeping Workspace as-is.")
            return

    scratch_path.mkdir(parents=True, exist_ok=True)

    # create environment with all necessary files
    # _start_subprocess(['git', 'clone', GIT_REPO, basedir])
    start_subprocess([SETENV_SH, "-N", workspace_path.name], cwd=basedir, ssh=ssh)

    # create autosix results folder.
    # Needs to be done after above command (as it crashes if folder exists)
    # but before end of this stage (as it needs to write the stagefile)
    get_autosix_results_path(jobname, basedir).mkdir(exist_ok=True, parents=True)


def _create_sixdeskenv(jobname: str, basedir: Path, **kwargs):
    """ Fills sixdeskenv mask and copies it to workspace """
    workspace_path = get_workspace_path(jobname, basedir)
    scratch_path = get_scratch_path(basedir)
    sixdeskenv_path = get_sixdeskenv_path(jobname, basedir)

    missing = [key for key in SIXENV_REQUIRED if key not in kwargs.keys()]
    if len(missing):
        raise ValueError(f"The following keys are required but missing {missing}.")

    sixenv_replace = SIXENV_DEFAULT.copy()
    sixenv_replace.update(kwargs)
    sixenv_replace.update(
        dict(
            JOBNAME=jobname,
            WORKSPACE=workspace_path.name,
            BASEDIR=str(basedir),
            SCRATCHDIR=str(scratch_path),
            TURNSPOWER=np.log10(sixenv_replace["TURNS"]),
        )
    )

    if any(sixenv_replace[key] is None for key in SEED_KEYS):
        for key in SEED_KEYS:
            sixenv_replace[key] = 0

    # the following checks are limits of SixDesk in 2020
    # and might be fixed upstream in the future
    if sixenv_replace["AMPMAX"] < sixenv_replace["AMPMIN"]:
        raise ValueError("Given AMPMAX is smaller than AMPMIN.")

    if (sixenv_replace["AMPMAX"] - sixenv_replace["AMPMIN"]) % sixenv_replace["AMPSTEP"]:
        raise ValueError("The amplitude range need to be dividable by the amplitude steps!")

    if not sixenv_replace["ANGLES"] % 2:
        raise ValueError("The number of angles needs to be an uneven one.")

    sixenv_text = SIXDESKENV_MASK.read_text()
    sixdeskenv_path.write_text(sixenv_text % sixenv_replace)
    LOG.debug("sixdeskenv written.")


def _create_sysenv(jobname: str, basedir: Path, binary_path: Path):
    """ Fills sysenv mask and copies it to workspace """
    LOG.info(f"Chosen binary for mask '{str(binary_path)}'")
    sysenv_path = get_sysenv_path(jobname, basedir)
    sysenv_replace = dict(
        MADXPATH=str(binary_path.parent),
        MADXBIN=binary_path.name,
    )
    sysenv_text = SYSENV_MASK.read_text()
    sysenv_path.write_text(sysenv_text % sysenv_replace)
    LOG.debug("sysenv written.")


def _write_mask(jobname: str, basedir: Path, mask_text: str, **kwargs):
    """ Fills mask with arguments and writes it out. """
    masks_path = get_masks_path(jobname, basedir)
    seed_range = [kwargs.get(key, SIXENV_DEFAULT[key]) for key in SEED_KEYS]

    if seed_range.count(None) == 1:
        raise ValueError(
            "First- or Lastseed is set, but the other one is deactivated. " "Set or unset both."
        )

    if ("%SEEDRAN" not in mask_text) and ("%SEEDRAN" not in kwargs.values()) and any(seed_range):
        raise ValueError(
            "First- and Lastseed are set, but no seed-variable '%SEEDRAN' found in mask."
        )

    mask_text = mask_text.replace("%SEEDRAN", "#!#SEEDRAN")  # otherwise next line will complain
    mask_filled = mask_text % kwargs
    mask_filled = mask_filled.replace(
        "#!#SEEDRAN", "%SEEDRAN"
    )  # bring seedran back for sixdesk seed-loop

    with open(masks_path / f"{jobname}.mask", "w") as mask_out:
        mask_out.write(mask_filled)
