import shutil
from pathlib import Path

import numpy as np
from omc3.utils import logging_tools

from pylhc.sixdesk_tools.utils import (MADX_PATH, SIXENV_REQUIRED, SIXENV_DEFAULT,
                                       SYSENV_MASK, SIXDESKENV_MASK, SETENV_SH,
                                       start_subprocess,
                                       get_sixjobs_path, get_workspace_path,
                                       get_scratch_path, get_masks_path, get_mad6t_mask_path)

LOG = logging_tools.get_logger(__name__)


# Main -------------------------------------------------------------------------

def create_jobs(jobname: str, basedir: Path, mask_text: str, **kwargs):
    binary_path = kwargs.pop('binary_path', MADX_PATH)
    ssh = kwargs.pop('ssh', None)

    sixjobs_path = get_sixjobs_path(jobname, basedir)
    _create_workspace(jobname, basedir, ssh=ssh)
    _create_sysenv(jobname, basedir, binary_path=binary_path)
    _create_sixdeskenv(jobname, basedir, **kwargs)
    _write_mask(jobname, basedir, mask_text, **kwargs)

    start_subprocess([SETENV_SH, '-s'], cwd=sixjobs_path, ssh=ssh)
    LOG.info("Workspace fully set up.")


def remove_twiss_fail_check(jobname: str, basedir: Path):
    """ Comments the "Twiss fail" check from mad6t.sh """
    LOG.info("Applying twiss-fail hack.")
    mad6t_path = get_mad6t_mask_path(jobname, basedir)
    with open(mad6t_path, 'r') as f:
        lines = f.readlines()

    check_started = False
    for idx, line in enumerate(lines):
        if line.startswith('grep -i "TWISS fail"'):
            check_started = True

        if check_started:
            lines[idx] = f'# {line}'
            if line.startswith('fi'):
                break
    else:
        LOG.info("'TWISS fail' not found in mad6t.sh")
        return

    with open(mad6t_path, 'w') as f:
        f.writelines(lines)


# Helper -----------------------------------------------------------------------


def _create_workspace(jobname: str, basedir: Path, ssh: str = None):
    """ Create workspace structure (with default files). """
    workspace_path, scratch_path = get_workspace_path(jobname, basedir), get_scratch_path(basedir)
    LOG.info(f'Creating new workspace in "{str(workspace_path)}"')

    if workspace_path.exists():
        LOG.warning(f'Workspace in "{str(workspace_path)}" already exists. ')
        LOG.info("Do you want to delete the old workspace? [y/N]")
        user_answer = input()
        if user_answer.lower().startswith('y'):
            shutil.rmtree(workspace_path)
            shutil.rmtree(scratch_path)
        else:
            LOG.warning("Keeping Workspace as-is.")
            return

    scratch_path.mkdir(parents=True, exist_ok=True)

    # create environment with all necessary files
    # _start_subprocess(['git', 'clone', GIT_REPO, basedir])
    start_subprocess([SETENV_SH, "-N", workspace_path.name], cwd=basedir, ssh=ssh)


def _create_sixdeskenv(jobname: str, basedir: Path, **kwargs):
    """ Fills sixdeskenv mask and copies it to workspace """
    workspace_path = get_workspace_path(jobname, basedir)
    scratch_path = get_scratch_path(basedir)
    sixjobs_path = get_sixjobs_path(jobname, basedir)

    missing = [key for key in SIXENV_REQUIRED if key not in kwargs.keys()]
    if len(missing):
        raise ValueError(f'The following keys are required but missing {missing}.')

    sixenv_replace = SIXENV_DEFAULT.copy()
    sixenv_replace.update(kwargs)
    sixenv_replace.update(dict(
        JOBNAME=jobname,
        WORKSPACE=workspace_path.name,
        BASEDIR=str(basedir),
        SCRATCHDIR=str(scratch_path),
        TURNSPOWER=np.log10(sixenv_replace['TURNS'])
    ))

    with open(SIXDESKENV_MASK, 'r') as f:
        sixenv_text = f.read()

    sixenv_text = sixenv_text % sixenv_replace

    with open(sixjobs_path / 'sixdeskenv', 'w') as f:
        f.write(sixenv_text)
    LOG.debug("sixdeskenv written.")


def _create_sysenv(jobname: str, basedir: Path, binary_path: Path = MADX_PATH):
    """ Fills sysenv mask and copies it to workspace """
    LOG.info(f"Chosen binary for mask '{str(binary_path)}'")
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    sysenv_replace = dict(
        MADXPATH=str(binary_path.parent),
        MADXBIN=binary_path.name,
    )

    with open(SYSENV_MASK, 'r') as f:
        sysenv_text = f.read()

    sysenv_text = sysenv_text % sysenv_replace

    with open(sixjobs_path / 'sysenv', 'w') as f:
        f.write(sysenv_text)

    LOG.debug("sysenv written.")


def _write_mask(jobname: str, basedir: Path, mask_text: str, **kwargs):
    """ Fills mask with arguments and writes it out. """
    masks_path = get_masks_path(jobname, basedir)
    seed_range = [kwargs.get(key, SIXENV_DEFAULT[key]) for key in ('FIRSTSEED', 'LASTSEED')]

    # seed_vars = re.findall(r'%\(?SEEDRAN\)?', mask_text)
    if ('%SEEDRAN' not in mask_text) and ('%SEEDRAN' not in kwargs.values()) and (seed_range[0] != seed_range[1]):
        raise ValueError("First and Lastseed are given, but no seed-variable '%SEEDRAN' found in mask.")

    mask_text = mask_text.replace('%SEEDRAN', '#!#SEEDRAN')  # otherwise next line will complain
    mask_filled = mask_text % kwargs
    mask_filled = mask_filled.replace('#!#SEEDRAN', '%SEEDRAN')  # bring seedran back for sixdesk seed-loop

    with open(masks_path / f'{jobname}.mask', 'w') as mask_out:
        mask_out.write(mask_filled)
