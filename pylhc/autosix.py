"""
AutoSix
-------------------

Wrapper for SixDesk to perform the setup and steps needed automatically.





:author: jdilly

"""
import itertools
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

from pylhc.constants.autosix import (STAGES, HEADER_BASEDIR, get_stagefile_path,
                                     DEFAULTS, SIXENV_REQUIRED, SIXENV_DEFAULT)
from pylhc.htc.mask import generate_jobdf_index
from pylhc.job_submitter import JOBSUMMARY_FILE, COLUMN_JOBID, check_replace_dict, keys_to_path
from pylhc.sixdesk_tools.create_workspace import create_jobs, remove_twiss_fail_check
from pylhc.sixdesk_tools.submit import (
    submit_mask, submit_sixtrack, check_sixtrack_input, check_sixtrack_output,
    sixdb_cmd, sixdb_load)
from pylhc.sixdesk_tools.utils import is_locked, check_mask, check_stage, StageSkip

LOG = logging_tools.get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="mask",
        type=PathOrStr,
        required=True,
        help="Program mask to use",
    )
    params.add_parameter(
        name="working_directory",
        type=PathOrStr,
        required=True,
        help="Directory where data should be put",
    )
    params.add_parameter(
        name="replace_dict",
        help=("Dict with keys of the strings to be replaced in the mask (required) "
              "as well as the mask_sixdeskenv and mask_sysenv files "
              "in the sixdesk_tools module. "
              f"Required fields are {', '.join(SIXENV_REQUIRED)}. "
              f"Optional fields are {', '.join(SIXENV_DEFAULT.keys())}. "
              "These keys can also be used in the mask if needed. "
              "The values of this dict are lists of values to replace "
              "these or single entries."),
        type=DictAsString,
        required=True
    )
    params.add_parameter(
        name="executable",
        default=DEFAULTS['executable'],
        type=PathOrStr,
        help="Path to executable.",
    )
    params.add_parameter(
        name="python",
        default=DEFAULTS['python'],
        type=PathOrStr,
        help="Path to python to use with sixdb (python3 with requirements installed).",
    )
    params.add_parameter(
        name="jobid_mask",
        help="Mask to name jobs from replace_dict",
        type=str,
    )
    params.add_parameter(
        name="ssh",
        help="Run htcondor from this machine via ssh (needs access to the `working_directory`)",
        type=str,
    )
    params.add_parameter(
        name="unlock",
        help="Forces unlocking of folders.",
        action="store_true",
    )
    params.add_parameter(
        name="ignore_twissfail_check",
        help=("Ignore the check for 'Twiss fail' in the submission file. "
              "This is a hack needed in case this check greps the wrong lines, "
              "e.g. in madx-comments. USE WITH CARE!!"),
        action="store_true",
    )
    params.add_parameter(
        name="resubmit",
        help="Resubmits if needed.",
        action="store_true",
    )
    params.add_parameter(
        name="da_turnstep",
        type=int,
        help="Step between turns used in DA-vs-Turns plot.",
        default=DEFAULTS['da_turnstep'],
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    """ Loop to create jobs from replace dict product matrix. """
    LOG.info("Starting autosix.")
    with open(opt.mask, 'r') as mask_f:
        mask = mask_f.read()
    opt = _check_opts(mask, opt)
    save_config(opt.working_directory, opt, __file__)

    jobdf = _generate_jobs(opt.working_directory, opt.jobid_mask, **opt.replace_dict)
    for job_args in jobdf.iterrows():
        setup_and_run(jobname=job_args[0],
                      basedir=opt.working_directory,
                      # kwargs:
                      ssh=opt.ssh,
                      python=opt.python,
                      unlock=opt.unlock,
                      resubmit=opt.resubmit,
                      da_turnstep=opt.da_turnstep,
                      ignore_twissfail_check=opt.ignore_twissfail_check,
                      # kwargs passed only to create_jobs:
                      mask_text=mask,
                      binary_path=opt.executable,
                      **job_args[1])


def setup_and_run(jobname: str, basedir: Path, **kwargs):
    """ Main submitting procedure for single job.

    Args:
        jobname (str): Name of the job/study
        basedir (Path): Working directory

    Keyword Args (optional):
        unlock (bool): unlock folder
        ssh (str): ssh-server to use
        python (str): python binary to use for sixDB
        resubmit(bool): Resubmit jobs if checks fail
        da_turnstep (int): Step in turns for DA
        ignore_twissfail_check (bool): Hack to ignore check for 'Twiss fail' after run

    Keyword Args (needed for create jobs):
        mask_text (str): Content of the mask to use.
        binary_path (Path): path to binary to use in jobs
        All Key=Values needed to fill the mask!
        All Key=Values needed to fill the mask!

    """
    LOG.info(f"vv---------------- Job {jobname} -------------------vv")
    unlock: bool = kwargs.pop('unlock', False)
    ssh: str = kwargs.pop('ssh', None)
    python: Union[Path, str] = kwargs.pop('python', DEFAULTS['python'])
    resubmit: bool = kwargs.pop('resubmit', False)
    da_turnstep: int = kwargs.pop('da_turnstep', DEFAULTS['da_turnstep'])
    ignore_twissfail_check: bool = kwargs.pop('ignore_twissfail_check', False)

    if is_locked(jobname, basedir, unlock=unlock):
        LOG.info(f"{jobname} is locked. Try 'unlock' flag if this causes errors.")

    with check_stage(STAGES.create_jobs, jobname, basedir) as check_ok:
        """ 
        create workspace
        > cd $basedir
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/set_env.sh -N workspace-$jobname
         
        write sixdeskenv, sysenv, filled mask (manual)
        
        initialize workspace
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/set_env.sh -s
         
        remove the twiss-fail check in sixtrack_input
        (manual)
        """
        if check_ok:
            create_jobs(jobname, basedir, ssh=ssh, **kwargs)
            if ignore_twissfail_check:  # Hack
                remove_twiss_fail_check(jobname, basedir)

    with check_stage(STAGES.submit_mask, jobname, basedir) as check_ok:
        """
        submit for input generation
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/mad6t.sh -s
        """
        if check_ok:
            submit_mask(jobname, basedir, ssh=ssh)
            return  # takes a while, so we interrupt here

    with check_stage(STAGES.check_input, jobname, basedir) as check_ok:
        """
        Check if input files have been generated properly
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/mad6t.sh -c
        
        If not, and resubmit is active
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/mad6t.sh -w
        """
        if check_ok:
            check_sixtrack_input(jobname, basedir, ssh=ssh, resubmit=resubmit)

    with check_stage(STAGES.submit_sixtrack, jobname, basedir) as check_ok:
        """
        Generate simulation files (-g) and check if runnable (-c) and submit (-s) (-g -c -s == -a).
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/run_six.sh -a
        """
        if check_ok:
            submit_sixtrack(jobname, basedir, ssh=ssh)
            return  # takes even longer

    with check_stage(STAGES.check_sixtrack_output, jobname, basedir) as check_ok:
        """
        Checks sixtrack output via run_status. If this fails even though all 
        jobs have finished on the scheduler, check the log-output (run_status
        messages are logged to debug).
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/run_status
        
        If not, and resubmit is active
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/run_six.sh -i
        """
        if check_ok:
            check_sixtrack_output(jobname, basedir, ssh=ssh, resubmit=resubmit)

    with check_stage(STAGES.sixdb_load, jobname, basedir) as check_ok:
        """
        Gather results into database via sixdb.
        > cd $basedir/workspace-$jobname/sixjobs
        > python3 /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/externals/SixDeskDB/sixdb . load_dir
        """
        if check_ok:
            sixdb_load(jobname, basedir, python=python, ssh=ssh)

    with check_stage(STAGES.sixdb_cmd, jobname, basedir) as check_ok:
        """
        Analysise results in database via sixdb.
        > cd $basedir/workspace-$jobname/sixjobs
        > python3 /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/externals/SixDeskDB/sixdb $jobname da 
        
        when fixed:
        > python3 /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/externals/SixDeskDB/sixdb $jobname da_vs_turns -turnstep 100 -outfile
        > python3 /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/externals/SixDeskDB/sixdb $jobname plot_da_vs_turns
        """
        if check_ok:
            pass
            sixdb_cmd(jobname, basedir, cmd=['da'], python=python, ssh=ssh)

            # da_vs_turns is broken at the moment (jdilly, 19.10.2020)
            # sixdb_cmd(jobname, basedir, cmd=['da_vs_turns', '-turnstep', str(da_turnstep), '-outfile'],
            #           python=python, ssh=ssh)
            # sixdb_cmd(jobname, basedir, cmd=['plot_da_vs_turns'], python=python, ssh=ssh)

    with check_stage(STAGES.final, jobname, basedir) as check_ok:
        """ Just info about finishing this script and where to check the stagefile. """
        if check_ok:
            stage_file = get_stagefile_path(jobname, basedir)
            LOG.info(f"All stages run. Check stagefile {str(stage_file)} "
                     "in case you want to rerun some stages.")
            raise StageSkip()

    LOG.info(f"^^---------------- Job {jobname} -------------------^^")

# Helper for main --------------------------------------------------------------


def _check_opts(mask_text, opt):
    opt = keys_to_path(opt, 'mask', 'working_directory', 'executable')
    check_mask(mask_text, opt.replace_dict)
    opt.replace_dict = check_replace_dict(opt.replace_dict)
    return opt


def _generate_jobs(basedir, jobid_mask, **kwargs) -> tfs.TfsDataFrame:
    """ Generates product matrix for job-values and stores it as TfsDataFrame. """
    LOG.debug("Creating Jobs")
    values_grid = np.array(list(itertools.product(*kwargs.values())), dtype=object)
    job_df = tfs.TfsDataFrame(
        headers={HEADER_BASEDIR: basedir},
        index=generate_jobdf_index(None, jobid_mask, kwargs.keys(), values_grid),
        columns=list(kwargs.keys()),
        data=values_grid,
    )
    try:
        # pandas >= 1.0 functionality with convert_dtypes
        job_df = job_df.convert_dtypes()
    except AttributeError:
        # fix for pandas < 1.0
        job_df = job_df.apply(partial(pd.to_numeric, errors='ignore'))
    tfs.write(basedir / JOBSUMMARY_FILE, job_df, save_index=COLUMN_JOBID)
    return job_df


if __name__ == '__main__':
    main()
