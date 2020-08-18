import itertools
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

from pylhc.htc.mask import generate_jobdf_index
from pylhc.job_submitter import JOBSUMMARY_FILE, COLUMN_JOBID, check_replace_dict
from pylhc.sixdesk_tools.create_workspace import create_jobs, remove_twiss_fail_check
from pylhc.sixdesk_tools.submit import submit_mask, submit_sixtrack, check_generated_input_files
from pylhc.sixdesk_tools.utils import is_locked, MADX_PATH, check_mask, HEADER_BASEDIR, STAGES, check_stage

LOG = logging_tools.get_logger(__name__, level_console=logging_tools.DEBUG, force_color=True)


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
        help="Dict containing the str to replace as keys and values a list of parameters to replace",
        type=DictAsString,
        required=True
    )
    params.add_parameter(
        name="executable",
        default=MADX_PATH,
        type=PathOrStr,
        help="Path to executable.",
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
        name="resubmit",
        help="Resubmits if needed.",
        action="store_true",
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting HTCondor Job-submitter.")
    with open(opt.mask, 'r') as mask_f:
        mask = mask_f.read()
    opt = _check_opts(mask, opt)
    save_config(opt.working_directory, opt, __file__)

    jobdf = _generate_jobs(opt.working_directory, opt.jobid_mask, **opt.replace_dict)
    for job_args in jobdf.iterrows():
        setup_and_run(jobname=job_args[0],
                      basedir=opt.working_directory,
                      mask=mask,
                      binary_path=opt.executable,
                      ssh=opt.ssh,
                      unlock=opt.unlock,
                      resubmit=opt.resubmit,
                      **job_args[1])


def setup_and_run(jobname, basedir, mask, **kwargs):
    """ Main submitting procedure for single job. """
    LOG.info(f"Job {jobname} -----------")
    unlock = kwargs.pop('unlock', False)
    ssh = kwargs.pop('ssh', None)
    resubmit = kwargs.pop('resubmit', False)

    if is_locked(jobname, basedir, unlock=unlock):
        LOG.info(f"{jobname} is locked. Try 'unlock' flag if this causes errors.")

    with check_stage(STAGES.create_jobs, jobname, basedir) as run:
        """ 
        create workspace
        > cd $basedir
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/set_env.sh -N workspace-$jobname
         
        write sixdeskenv, sysenv, filled mask
        (manual)
         
        initialize workspace
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/set_env.sh -s
         
        remove the twiss-fail check in sixtrack_input
        (manual)
        """
        if run:
            create_jobs(jobname, basedir, mask, ssh=ssh, **kwargs)
            remove_twiss_fail_check(jobname, basedir)  # Hack

    with check_stage(STAGES.submit_mask, jobname, basedir) as run:
        """
        submit for input generation
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/mad6t.sh -s
        """
        if run:
            submit_mask(jobname, basedir, ssh=ssh)  # takes a while

    with check_stage(STAGES.check_input, jobname, basedir) as run:
        """
        Check if input files have been generated properly
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/mad6t.sh -c
        
        If not, resubmit
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/mad6t.sh -w
        """
        if run:
            check_generated_input_files(jobname, basedir,
                                        ssh=ssh, resubmit=resubmit)

    with check_stage(STAGES.submit_sixtrack, jobname, basedir) as run:
        """
        Generate simulation files (-g) and check if runnable (-c) and submit (-s) (-g -c -s == -a).
        > cd $basedir/workspace-$jobname/sixjobs
        > /afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/run_six.sh -a
        """
        if run:
            submit_sixtrack(jobname, basedir, ssh=ssh)  # takes even longer


# Helper for main --------------------------------------------------------------


def _check_opts(mask_text, opt):
    check_mask(mask_text, opt.replace_dict)
    opt.replace_dict = check_replace_dict(opt.replace_dict)
    return opt


def _generate_jobs(basedir, jobid_mask, **kwargs):
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
    main(
        working_directory=Path('/afs/cern.ch/work/j/jdilly/sixdeskbase'),
        mask=Path('/home/jdilly/Work/study.20.irnl_correction_with_feeddown/code.full_study/python_mask.py'),
        executable=Path('/afs/cern.ch/work/j/jdilly/public/venvs/for_htc/bin/python'),
        ssh='lxplus',
        replace_dict=dict(
            BEAM=1,
            TURNS=100000,
            AMPMIN=2, AMPMAX=20, AMPSTEP=5,
            ANGLES=50,
            B6viaB4=False,
            SEED='%SEEDRAN'
        ),
        jobid_mask="B%(BEAM)d_B6viaB4_%(B6viaB4)s",
        # unlock=True,
        # resubmit=True,
    )

