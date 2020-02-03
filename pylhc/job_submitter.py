"""
MADX Job-Submitter
--------------------

Allows to execute a parametric madx study using a madx mask and a dictionary with parameters to replace.
Parameters to be replaced must be present in mask as `%(PARAMETER)s`
(other types apart from string also allowed).

When submitting to HTCondor, madx data to be transferred back to the working directory
must be written in a sub-folder defined by `job_output_directory` which defaults to `Outputdata`.

Script also allows to check if all htcondor job finished successfully,
resubmissions with a different parameter grid, and local execution.

A `Jobs.tfs` is created in the working directory containing the Job Id,
parameter per job and job directory for further post processing.

*--Required--*

- **mask** *(str)*: Madx mask to use

- **replace_dict** *(DictAsString)*: Dict containing the str to replace as
  keys and values a list of parameters to replace

- **working_directory** *(str)*: Directory where data should be put


*--Optional--*

- **append_jobs**: Flag to rerun job with finer/wider grid,
  already existing points will not be reexecuted.

  Action: ``store_true``
- **check_files** *(str)*: List of files/file-name-masks expected to be in the
  'job_output_dir' after a successful job (for appending/resuming). Uses the 'glob'
  function, so unix-wildcards (*) are allowed. If not given, only the presence of the folder itself is checked.
- **dryrun**: Flag to only prepare folders and scripts,
  but does not start madx/submit jobs.
  Together with `resume_jobs` this can be use to check which jobs succeeded and which failed.

  Action: ``store_true``
- **executable** *(str)*: Path to executable or job-type (of ['madx', 'python3', 'python2']) to use.

- **htc_arguments** *(DictAsString)*: Additional arguments for htcondor, as Dict-String.
  Choices: ['accounting_group', 'max_retries', 'notification', 'priority']

  Default: ``{}``
- **job_output_dir** *(str)*: The name of the output dir of the job. (Make sure your script puts its data there!)

  Default: ``Outputdata``
- **jobflavour** *(str)*: Jobflavour to give rough estimate of runtime of one job

  Choices: ``('espresso', 'microcentury', 'longlunch', 'workday', 'tomorrow', 'testmatch', 'nextweek')``
  Default: ``workday``
- **jobid_mask** *(str)*: Mask to name jobs from replace_dict

- **num_processes** *(int)*: Number of processes to be used if run locally

  Default: ``4``
- **resume_jobs**: Only do jobs that did not work.

  Action: ``store_true``
- **run_local**: Flag to run the jobs on the local machine. Not suggested.

  Action: ``store_true``
- **script_arguments** *(DictAsString)*: Additional arguments to pass to the script,
  as dict in key-value pairs ('--' need to be included in the keys).

  Default: ``{}``

:module: madx_submitter
:author: mihofer, jdilly

"""
import itertools
import multiprocessing
import os
import subprocess
import re
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import tfs
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import save_options_to_config
from generic_parser.tools import print_dict_tree

import pylhc.htc.utils as htcutils
import pylhc.madx.mask as mask_processing
from pylhc.htc.utils import (COLUMN_SHELL_SCRIPT, COLUMN_JOB_DIRECTORY,
                             JOBFLAVOURS, HTCONDOR_JOBLIMIT, EXECUTEABLEPATH)

from omc3.utils import logging_tools


JOBSUMMARY_FILE = 'Jobs.tfs'
JOBDIRECTORY_PREFIX = 'Job'
COLUMN_JOBID = "JobId"
CONFIG_FILE = 'config.ini'


LOG = logging_tools.get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="mask",
        type=str,
        required=True,
        help="Madx mask to use",
    )
    params.add_parameter(
        name="working_directory",
        type=str,
        required=True,
        help="Directory where data should be put",
    )
    params.add_parameter(
        name="executable",
        default='madx',
        type=str,
        help="Path to executable or job-type "
             f"(of {str(list(EXECUTEABLEPATH.keys()))}) to use.",
    )
    params.add_parameter(
        name="jobflavour",
        type=str,
        choices=JOBFLAVOURS,
        default='workday',
        help="Jobflavour to give rough estimate of runtime of one job ",
    )
    params.add_parameter(
        name="run_local",
        action="store_true",
        help="Flag to run the jobs on the local machine. Not suggested.",
    )
    params.add_parameter(
        name="resume_jobs",
        action="store_true",
        help="Only do jobs that did not work.",
    )
    params.add_parameter(
        name="append_jobs",
        action="store_true",
        help="Flag to rerun job with finer/wider grid, already existing points will not be reexecuted.",
    )
    params.add_parameter(
        name="dryrun",
        action="store_true",
        help="Flag to only prepare folders and scripts, but does not start madx/submit jobs. "
             "Together with `resume_jobs` this can be use to check which jobs "
             "succeeded and which failed.",
    )
    params.add_parameter(
        name="replace_dict",
        help="Dict containing the str to replace as keys and values a list of parameters to replace",
        type=DictAsString,
        required=True
    )
    params.add_parameter(
        name="script_arguments",
        help="Additional arguments to pass to the script, as dict in key-value pairs "
             "('--' need to be included in the keys).",
        type=DictAsString,
        default={},
    )
    params.add_parameter(
        name="num_processes",
        help="Number of processes to be used if run locally",
        type=int,
        default=4
    )
    params.add_parameter(
        name="check_files",
        help=("List of files/file-name-masks expected to be in the "
              "'job_output_dir' after a successful job "
              "(for appending/resuming). Uses the 'glob' function, so "
              "unix-wildcards (*) are allowed. If not given, only the "
              "presence of the folder itself is checked."
              ),
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="jobid_mask",
        help="Mask to name jobs from replace_dict",
        type=str,
    )
    params.add_parameter(
        name="job_output_dir",
        help="The name of the output dir of the job. (Make sure your script puts its data there!)",
        type=str,
        default="Outputdata"
    )
    params.add_parameter(
        name="htc_arguments",
        help="Additional arguments for htcondor, as Dict-String. "
             "Choices: ['accounting_group', 'max_retries', 'notification', 'priority']",
        type=DictAsString,
        default={}
    )

    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting MADX-submitter.")
    opt = _check_opts(opt)
    opt = _convert_to_paths(opt, ('working_directory', 'job_output_dir', 'mask'))
    save_options_to_config(opt.working_directory / CONFIG_FILE, opt)

    job_df = _create_jobs(opt.working_directory, opt.mask, opt.jobid_mask, opt.replace_dict,
                          opt.job_output_dir, opt.append_jobs, opt.executable, opt.script_arguments)
    job_df, dropped_jobs = _drop_already_run_jobs(job_df, opt.resume_jobs or opt.append_jobs,
                                             opt.job_output_dir, opt.check_files)

    if not opt.dryrun:
        _run(job_df, opt.working_directory, opt.job_output_dir,
             opt.jobflavour, opt.num_processes, opt.run_local, opt.htc_arguments)
    else:
        _print_stats(job_df.index, dropped_jobs)


# Main Functions ---------------------------------------------------------------


def _create_jobs(cwd, maskfile, jobid_mask, replace_dict, output_dir, append_jobs, executable, script_args):
    LOG.debug("Creating MADX-Jobs")
    values_grid = np.array(list(itertools.product(*replace_dict.values())), dtype=object)

    if append_jobs:
        jobfile_path = cwd / JOBSUMMARY_FILE
        try:
            job_df = tfs.read(str(jobfile_path.absolute()), index=COLUMN_JOBID)
        except FileNotFoundError:
            raise FileNotFoundError("Cannot append jobs, as no previous jobfile was found at "
                                    f"'{jobfile_path}'")
        mask = [elem not in job_df[replace_dict.keys()].values for elem in values_grid]
        njobs = mask.count(True)
        values_grid = values_grid[mask]

    else:
        njobs = len(values_grid)
        job_df = tfs.TfsDataFrame()

    if njobs == 0:
        raise ValueError(f'No (new) jobs found!')
    if njobs > HTCONDOR_JOBLIMIT:
        raise ValueError(f'Too many jobs! Allowed {HTCONDOR_JOBLIMIT}, given {njobs}.')

    LOG.debug(f'Initial number of jobs: {njobs:d}')

    data_df = pd.DataFrame(
        index=_generate_index(job_df, jobid_mask, replace_dict.keys(), values_grid),
        columns=list(replace_dict.keys()),
        data=values_grid)
    job_df = job_df.append(data_df, sort=False)

    job_df = _setup_folders(job_df, cwd)

    # creating all madx jobs
    job_df = mask_processing.create_madx_jobs_from_mask(job_df, maskfile, replace_dict.keys())

    # creating all shell scripts
    job_df = htcutils.write_bash(job_df, output_dir, executable=executable, cmdline_arguments=script_args)

    job_df = _set_auto_tfs_column_types(job_df)
    tfs.write(str(cwd / JOBSUMMARY_FILE), job_df, save_index=COLUMN_JOBID)
    return job_df


def _drop_already_run_jobs(job_df, drop_jobs, output_dir, check_files):
    LOG.debug("Dropping already finished jobs, if necessary.")
    finished_jobs = []
    if drop_jobs:
        finished_jobs = [idx for idx, row in job_df.iterrows()
                         if _job_was_successful(row, output_dir, check_files)]
        LOG.info(f"{len(finished_jobs):d} of {len(job_df.index):d}"
                 " Jobs have already finished and will be skipped.")
        job_df = job_df.drop(index=finished_jobs)
    return job_df, finished_jobs


def _run(job_df, cwd, output_dir, flavour, num_processes, run_local, additional_htc_arguments):
    if run_local:
        LOG.info(f"Running {len(job_df.index)} jobs locally in {num_processes:d} processes.")
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(_execute_shell, job_df.iterrows())

    else:
        LOG.info(f"Submitting {len(job_df.index)} jobs on htcondor, flavour '{flavour}'.")
        # create submission file
        subfile = htcutils.make_subfile(cwd, job_df, output_dir=output_dir, duration=flavour,
                                        **additional_htc_arguments)
        # submit to htcondor
        htcutils.submit_jobfile(subfile)


# Sub Functions ----------------------------------------------------------------


def _setup_folders(job_df, working_directory):
    LOG.debug("Setting up folders: ")
    job_df[COLUMN_JOB_DIRECTORY] = list(map(_return_job_dir,
                                            zip([working_directory] * len(job_df), job_df.index)))
    for job_dir in job_df[COLUMN_JOB_DIRECTORY]:
        try:
            os.mkdir(job_dir)
        except IOError:
            LOG.debug(f"   created '{job_dir}'.")
        else:
            LOG.debug(f"   failed '{job_dir}' (might already exist).")
    return job_df


def _job_was_successful(job_row, output_dir, files):
    output_dir = Path(job_row[COLUMN_JOB_DIRECTORY]) / output_dir
    success = output_dir.is_dir()
    if success and files is not None and len(files):
        for f in files:
            success &= len(list(output_dir.glob(f))) > 0
    return success


def _generate_index(old_df, mask, keys, values):
    if not mask:
        nold = len(old_df.index)
        start = nold-1 if nold > 0 else 0
        return range(start, start + values.shape[0])
    return [mask % dict(zip(keys, v)) for v in values]


def _return_job_dir(inp):
    return os.path.join(inp[0], f'{JOBDIRECTORY_PREFIX}.{inp[1]}')


def _execute_shell(df_row):
    idx, column = df_row
    with open(os.path.join(column[COLUMN_JOB_DIRECTORY], 'log.tmp'), 'w') as logfile:
        process = subprocess.Popen(['sh', column[COLUMN_SHELL_SCRIPT]],
                                   shell=False,
                                   stdout=logfile,
                                   stderr=subprocess.STDOUT,
                                   cwd=column[COLUMN_JOB_DIRECTORY])

    status = process.wait()
    return status


def _check_opts(opt):
    LOG.debug("Checking options")
    if opt.resume_jobs and opt.append_jobs:
        raise ValueError('Select either Resume jobs or Append jobs')

    with open(opt.mask, "r") as inputmask:  # checks that mask and dir are there
        mask = inputmask.read()

    dict_keys = set(opt.replace_dict.keys())
    mask_keys = _find_named_variables_in_mask(mask)
    not_in_mask = dict_keys - mask_keys
    not_in_dict = mask_keys - dict_keys

    if len(not_in_dict):
        raise KeyError("The following keys in the mask were not found in the given replace_dict: "
                       f"{str(not_in_dict).strip('{}')}")

    if len(not_in_mask):
        LOG.warning("The following replace_dict keys were not found in the given mask: "
                    f"{str(not_in_mask).strip('{}')}")

        # remove all keys which are not present in mask (otherwise unnecessary jobs)
        [opt.replace_dict.pop(key) for key in not_in_mask]
        if len(opt.replace_dict) == 0:
            raise KeyError('Empty replace-dictionary')

    print_dict_tree(opt, name="Input parameter", print_fun=LOG.debug)

    return opt


def _convert_to_paths(opt, keys):
    """ Converts strings (defined by keys) that should be paths to Path. """
    for key in keys:
        opt[key] = Path(opt[key])
    return opt


def _print_stats(new_jobs, finished_jobs):
    """ Print some quick statistics. """
    LOG.info("------------- QUICK STATS ----------------")
    LOG.info(f"Jobs total:{len(new_jobs) + len(finished_jobs):d}")
    LOG.info(f"Jobs to run: {len(new_jobs):d}")
    LOG.info(f"Jobs already finished: {len(finished_jobs):d}")
    LOG.info("---------- JOBS TO RUN NAMES -------------")
    for job_name in new_jobs:
        LOG.info(job_name)
    LOG.info("--------- JOBS FINISHED NAMES ------------")
    for job_name in finished_jobs:
        LOG.info(job_name)


def _set_auto_tfs_column_types(df):
    return df.apply(partial(pd.to_numeric, errors='ignore'))


def _find_named_variables_in_mask(mask):
    return set(re.findall(r"%\((\w+)\)", mask))


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
