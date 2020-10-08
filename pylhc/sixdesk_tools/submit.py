from pathlib import Path
from omc3.utils import logging_tools
from pylhc.sixdesk_tools.utils import (RUNSIX_SH, MAD_TO_SIXTRACK_SH,
                                       get_sixjobs_path, start_subprocess,
                                       StageSkip, get_workspace_path, RUNSTATUS_SH)

LOG = logging_tools.get_logger(__name__)


def submit_mask(jobname: str, basedir: Path, ssh: str = None):
    """ Run the mask (probably Madx) and generate sixtrack input files. """
    LOG.info("Submitting mask to run for sixtrack input generation.")
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    start_subprocess([MAD_TO_SIXTRACK_SH, '-s'], cwd=sixjobs_path, ssh=ssh)
    LOG.info("Submitted mask-jobs.")


def check_sixtrack_input(jobname: str, basedir: Path, ssh: str = None, resubmit: bool = False):
    """ Checks the generated input files needed by sixtrack and resubmits, if requested. """
    LOG.info("Checking if input files are present.")
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    try:
        start_subprocess([MAD_TO_SIXTRACK_SH, '-c'], cwd=sixjobs_path, ssh=ssh)
    except OSError as e:
        if resubmit:
            LOG.info("Resubmitting mask to run wrong seeds for sixtrack input generation.")
            start_subprocess([MAD_TO_SIXTRACK_SH, '-w'], cwd=sixjobs_path, ssh=ssh)
            raise StageSkip("Resubmitted input generation jobs.")
        else:
            raise StageSkip("Checking input files failed. Check (debug-) logs. "
                            "Maybe restart with 'resubmit' flag.") from e
    else:
        LOG.info("Check for input files was successful.")


def submit_sixtrack(jobname: str, basedir: Path, ssh: str = None, resubmit=False):
    """ Generate simulation files and check if runnable and submit. """
    LOG.info("Submitting to sixtrack.")
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    try:
        start_subprocess([RUNSIX_SH, '-a'], cwd=sixjobs_path, ssh=ssh)  # throws OSError if failed
    except OSError as e:
        raise StageSkip(f'Submit to sixtrack for {jobname} ended in error.'
                        f' Input generation possibly not finished. Check your Scheduler.') from e
    else:
        LOG.info("Submitted jobs to Sixtrack")


def check_sixtrack_output(jobname: str, basedir: Path, ssh: str = None, resubmit: bool = False):
    LOG.info("Checking if sixtrack has finished.")
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    try:
        start_subprocess([RUNSTATUS_SH], cwd=sixjobs_path, ssh=ssh)
    except OSError as e:
        raise StageSkip(f'Sixtrack for {jobname} seems to be incomplete.'
                        f'Run possibly not finished. Check (debug-) log or your Scheduler.') from e
    else:
        LOG.info("Sixtrack results are all present.")
