from pathlib import Path

from pylhc.sixdesk_tools.utils import (RUNSIX_SH, MAD_TO_SIXTRACK_SH,
                                       STAGES, stage_function,
                                       get_sixjobs_path, start_subprocess,)


@stage_function(STAGES.submit_mask)
def submit_mask(jobname: str, basedir: Path, ssh: str = None):
    """ Run the mask (probably Madx) and generate sixtrack input files. """
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    start_subprocess([MAD_TO_SIXTRACK_SH, '-s'], cwd=sixjobs_path, ssh=ssh)


@stage_function(STAGES.submit_sixtrack)
def submit_sixtrack(jobname: str, basedir: Path, ssh: str = None):
    """ Generate simulation files (-g) and check if runnable (-c) and submit (-s).

    > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/run_six.sh -g -c -s
    or (shorthand)
    > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/run_six.sh -a
    """
    sixjobs_path = get_sixjobs_path(jobname, basedir)
    start_subprocess([RUNSIX_SH, '-a'], cwd=sixjobs_path, ssh=ssh)  # throws OSError if failed


