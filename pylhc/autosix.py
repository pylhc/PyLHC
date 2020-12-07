"""
AutoSix
-------

A wrapper for SixDesk to perform the setup and steps needed automatically.

The idea is, to call this script with the same parameters and it runs through
all the steps automatically.

The functionality is similar to the :mod:`pylhc.job_submitter` in that
the inner product of a ``replace_dict`` is used to automatically create a set of
job-directories to gather the data.
To avoid conflicts each of these job-directories is a SixDesk workspace,
meaning there will be only one "study" per directory, .

The ``replace_dict`` contains variables for your mask as well as variables
for the SixDesk environment. See the description of ``replace_dict`` below.
In any other way, these __special__ variables behave like normal variables and
can also be inserted in your mask. They are also looped over in the same manner
as any other variable (if given as a list).

.. important::
    As the loop over Seeds is handled by SixDesk you need to set
    `FIRSTSEED` and `LASTSEED` to None or 0 to deactivate this loop.
    Otherwise a ``%SEEDRAN`` placeholder is required in your mask,
    which needs to be present **after** filling in the variables (see example below).


.. note::
    Unlike in the :mod:`pylhc.job_submitter`, the output directory of the
    HTCondor job (the 'mask-job') is not automatically transferred to the
    workspace. To have access to this data, you will need to specify different
    output-directories in your mask manually, e.g. using strings containing
    the variable placeholders.


.. code-block:: python

    from pathlib import Path
    from omc3.utils import logging_tools
    from pylhc import autosix

    LOG = logging_tools.get_logger(__name__)

    if __name__ == '__main__':
        autosix.main(
            working_directory=Path('/afs/cern.ch/work/u/user/sixdeskbase'),
            mask=Path('my_madx.mask'), # can contain any of the parameters used in replace_dict
            python=Path('/afs/cern.ch/work/u/user/my_venv/bin/python'),
            ignore_twissfail_check=False,  # if script prints 'check if twiss failed' or similar
            replace_dict=dict(
                # Part of the sixdesk-environment:
                TURNS=100000,
                AMPMIN=4, AMPMAX=30, AMPSTEP=2,
                ANGLES=11,
                # Examples for mask:
                BEAM=[1, 4],
                TUNE=[62.29, 62.30, 62.31, 62.31]
                OUTPUT='/afs/cern.ch/work/u/user/study_output/',
                SEED='%SEEDRAN',  # puts '%SEEDRAN' in the mask and lets sixdesk handle this loop
            ),
            jobid_mask="B%(BEAM)d-QX%(TUNE)s",
            # unlock=True,
            # resubmit=True,
            # ssh='lxplus.cern.ch',
        )


Upon running the script the job-matrix is created
(see __Jobs.tfs__ in working directory) and the following stages are run per job:

- ``create_jobs``: create workspace; fill sysenv, sixdeskenv, mask; initialize workspace.
- ``submit_mask``: submit mask-job to HTCondor. (__interrupt__)
- ``check_input``: check if sixdesk input is complete.
- ``submit_sixtrack``: submit sixdesk-jobs to HTCondor. (__interrupt__)
- ``check_sixtrack_output``: check if all sixdesk jobs are completed.
- ``sixdb_load``: crate database and load jobs output.
- ``sixdb_cmd``: calculated DA from database data.
- ``post_process``: extract data from database, write into _.tfs_ and plot.
- ``final``: announce everything has finihed


To keep track of the stages, they are written into the __stages\_completed.txt__
in the __autosix\_output__ directory in the workspaces.
Stages that are written in this file are assumed to be done and will be skipped.
To rerun a stage, delete the stage and all following stages in that file and
start your script anew.

The stages are run independently of each job, meaning different jobs can be
at different stages. E.g if one job has all data for the ``six_db`` analysis
already, but the others are still running on sixdesk the
``check_sixtrack_output`` stage will fail for these jobs but the other one will
just continue.

Because the stages after ``submit_mask`` and ``submit_sixtrack`` need only be
run after the jobs on HTCondor are completed, these two stages interrupt the
execution of stages if they have successfully finished. Check your scheduler
via ``condor_q`` and run your script again after everything is done, to
have autosix continue its work.


For the creation of polar plots, the function
:func:`pylhc.sixdesk_tools.post_process_da.plot_polar` is available, which is
used for the basic polar plotting in the ``post_process`` stage, but provides
more customization features if called manually.


Arguments:

*--Required--*

- **mask** *(PathOrStr)*:

    Program mask to use


- **replace_dict** *(DictAsString)*:

    Dict with keys of the strings to be replaced in the mask (required) as
    well as the mask_sixdeskenv and mask_sysenv files in the sixdesk_tools
    module. Required fields are TURNS, AMPMIN, AMPMAX, AMPSTEP,
    ANGLES. Optional fields are RESUBMISSION, PLATFORM, LOGLEVEL,
    FIRSTSEED, LASTSEED, ENERGY, NPAIRS, EMITTANCE, DIMENSIONS, WRITEBINS.
    These keys can also be used in the mask if needed. The values of this
    dict are lists of values to replace these or single entries.


- **working_directory** *(PathOrStr)*:

    Directory where data should be put


*--Optional--*

- **da_turnstep** *(int)*:

    Step between turns used in DA-vs-Turns plot.

    default: ``100``


- **executable** *(PathOrStr)*:

    Path to executable.

    default: ``/afs/cern.ch/user/m/mad/bin/madx``


- **ignore_twissfail_check**:

    Ignore the check for 'Twiss fail' in the submission file. This is a
    hack needed in case this check greps the wrong lines, e.g. in madx-
    comments. USE WITH CARE!!

    action: ``store_true``


- **jobid_mask** *(str)*:

    Mask to name jobs from replace_dict


- **python** *(PathOrStr)*:

    Path to python to use with sixdb (python3 with requirements
    installed).

    default: ``python3``


- **resubmit**:

    Resubmits if needed.

    action: ``store_true``


- **ssh** *(str)*:

    Run htcondor from this machine via ssh (needs access to the
    `working_directory`)


- **unlock**:

    Forces unlocking of folders.

    action: ``store_true``



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

from pylhc.constants.autosix import (
    STAGES,
    HEADER_BASEDIR,
    get_stagefile_path,
    DEFAULTS,
    SIXENV_REQUIRED,
    SIXENV_DEFAULT,
    get_autosix_results_path,
)
from pylhc.htc.mask import generate_jobdf_index
from pylhc.job_submitter import (
    JOBSUMMARY_FILE,
    COLUMN_JOBID,
    check_replace_dict,
    keys_to_path,
)
from pylhc.sixdesk_tools.create_workspace import create_jobs, remove_twiss_fail_check
from pylhc.sixdesk_tools.post_process_da import post_process_da
from pylhc.sixdesk_tools.submit import (
    submit_mask,
    submit_sixtrack,
    check_sixtrack_input,
    check_sixtrack_output,
    sixdb_cmd,
    sixdb_load,
)
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
        help=(
            "Dict with keys of the strings to be replaced in the mask (required) "
            "as well as the mask_sixdeskenv and mask_sysenv files "
            "in the sixdesk_tools module. "
            f"Required fields are {', '.join(SIXENV_REQUIRED)}. "
            f"Optional fields are {', '.join(SIXENV_DEFAULT.keys())}. "
            "These keys can also be used in the mask if needed. "
            "The values of this dict are lists of values to replace "
            "these or single entries."
        ),
        type=DictAsString,
        required=True,
    )
    params.add_parameter(
        name="executable",
        default=DEFAULTS["executable"],
        type=PathOrStr,
        help="Path to executable.",
    )
    params.add_parameter(
        name="python",
        default=DEFAULTS["python"],
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
        help=(
            "Ignore the check for 'Twiss fail' in the submission file. "
            "This is a hack needed in case this check greps the wrong lines, "
            "e.g. in madx-comments. USE WITH CARE!!"
        ),
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
        default=DEFAULTS["da_turnstep"],
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    """ Loop to create jobs from replace dict product matrix. """
    LOG.info("Starting autosix.")
    with open(opt.mask, "r") as mask_f:
        mask = mask_f.read()
    opt = _check_opts(mask, opt)
    save_config(opt.working_directory, opt, __file__)

    jobdf = _generate_jobs(opt.working_directory, opt.jobid_mask, **opt.replace_dict)
    for job_args in jobdf.iterrows():
        setup_and_run(
            jobname=job_args[0],
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
            **job_args[1],
        )


def setup_and_run(jobname: str, basedir: Path, **kwargs):
    """Main submitting procedure for single job.

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
    unlock: bool = kwargs.pop("unlock", False)
    ssh: str = kwargs.pop("ssh", None)
    python: Union[Path, str] = kwargs.pop("python", DEFAULTS["python"])
    resubmit: bool = kwargs.pop("resubmit", False)
    da_turnstep: int = kwargs.pop("da_turnstep", DEFAULTS["da_turnstep"])
    ignore_twissfail_check: bool = kwargs.pop("ignore_twissfail_check", False)

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
            sixdb_cmd(jobname, basedir, cmd=["da"], python=python, ssh=ssh)

            # da_vs_turns is broken at the moment (jdilly, 19.10.2020)
            # sixdb_cmd(jobname, basedir, cmd=['da_vs_turns', '-turnstep', str(da_turnstep), '-outfile'],
            #           python=python, ssh=ssh)
            # sixdb_cmd(jobname, basedir, cmd=['plot_da_vs_turns'], python=python, ssh=ssh)

    with check_stage(STAGES.post_process, jobname, basedir) as check_ok:
        """
        Extracts the analysed data in the database and writes them to three tfs files:

        - All DA values
        - Statistics over angles, listed per seed (+ Seed 0 as over seeds and angles)
        - Statistics over seeds, listed per angle

        The statistics over the seeds are then plotted in a polar plot.
        All files are outputted to the ``sixjobs/autosix_output`` folder in the job directory.
        """
        if check_ok:
            post_process_da(jobname, basedir)

    with check_stage(STAGES.final, jobname, basedir) as check_ok:
        """ Just info about finishing this script and where to check the stagefile. """
        if check_ok:
            stage_file = get_stagefile_path(jobname, basedir)
            LOG.info(
                f"All stages run. Check stagefile {str(stage_file)} "
                "in case you want to rerun some stages."
            )
            raise StageSkip()

    LOG.info(f"^^---------------- Job {jobname} -------------------^^")


# Helper for main --------------------------------------------------------------


def _check_opts(mask_text, opt):
    opt = keys_to_path(opt, "mask", "working_directory", "executable")
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
    tfs.write(basedir / JOBSUMMARY_FILE, job_df, save_index=COLUMN_JOBID)
    return job_df


if __name__ == "__main__":
    main()
