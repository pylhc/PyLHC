"""
MADX Job-Submitter
--------------------

Allows to execute a parametric madx study using a madx mask and a dictionary with parameters to replace.
Parameters to be replaced must be present in mask as %(PARAMETER)s.
When submitting to HTCondor, madx data to be transfered back to job_directory must be written in folder Outputdata.
Script also allows to check if all htcondor job finished successfully, resubmissions with a different parameter grid, and local excution.
Jobs.tfs is created in the working directory containing the Job Id, parameter per job and job directory for further post processing.

required arguments:
    --mask                  path to the madx mask to be used e.g. ./job_lhc_bbeat_misalign.mask
    --working_directory     directory where job directories will be put and mask is located
    --replace_dict          dictionary with keys being the parameters to be replaced in the mask and values the parameter values, e.g. {'PARAMETER_A':[1,2], 'PARAMETER_B':['a','b']}

optional arguments:
    --jobflavour           string giving upper limit on how long job can take, only used when submitting to HTCondor
    --local                flag to trigger local processing using the number of processes specified in --num_processes
    --num_processes        number of processes used when run locally
    --resume_jobs          flag which enables a check if all htcondor jobs finished successfully and resubmitting failed jobs
    --append_jobs          flag allowing to resubmit study with different replace_dict, rechecks if datapoints have already been executed in a previous study and only resubmits new jobs


:module: madx_submitter
:author: mihofer, jdilly

"""
import itertools
import multiprocessing
import os
import subprocess
import re
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
from pylhc.htc.utils import COLUMN_SHELL_SCRIPT, COLUMN_JOB_DIRECTORY
from pylhc.htc.utils import JOBFLAVOURS, HTCONDOR_JOBLIMIT
from pylhc.omc3.omc3.utils import logging_tools

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
        name="replace_dict",
        help="Dict containing the str to replace as keys and values a list of parameters to replace",
        type=DictAsString,
        required=True
    )
    params.add_parameter(
        name="num_processes",
        help="number of processes to be used if run locally",
        type=int,
        default=4
    )
    params.add_parameter(
        name="check_files",
        help="Files to check to count job as successfull",
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
        name="additional_parameters",
        help="Additional parameters for the job, as Dict-String. Choices: group, retries, notification, priority",
        type=DictAsString,
        default={}
    )

    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting MADX-submitter.")
    opt = _check_opts(opt)
    save_options_to_config(os.path.join(opt.working_directory, CONFIG_FILE), opt)

    job_df = _create_jobs(opt.working_directory, opt.mask, opt.jobid_mask, opt.replace_dict,
                          opt.job_output_dir, opt.append_jobs)
    job_df = _drop_already_run_jobs(job_df, opt.resume_jobs or opt.append_jobs,
                                    opt.job_output_dir, opt.check_files)

    _run(job_df, opt.working_directory, opt.job_output_dir,
         opt.jobflavour, opt.num_processes, opt.run_local, opt.additional_parameters)


# Main Functions ---------------------------------------------------------------


def _create_jobs(cwd, maskfile, jobid_mask, replace_dict, output_dir, append_jobs):
    LOG.debug("Creating MADX-Jobs")
    values_grid = np.array(list(itertools.product(*replace_dict.values())), dtype=object)

    if append_jobs:
        jobfile_path = os.path.join(cwd, JOBSUMMARY_FILE)
        try:
            job_df = tfs.read(jobfile_path, index=COLUMN_JOBID)
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
    job_df = htcutils.write_bash(job_df, output_dir, jobtype='madx')

    job_df = _set_auto_tfs_column_types(job_df)
    tfs.write(os.path.join(cwd, JOBSUMMARY_FILE), job_df, save_index=COLUMN_JOBID)
    return job_df


def _drop_already_run_jobs(job_df, drop_jobs, output_dir, check_files):
    LOG.debug("Dropping already finished jobs, if necessary.")
    if drop_jobs:
        finished_jobs = [idx for idx, row in job_df.iterrows() if _job_was_successful(row, output_dir, check_files)]
        LOG.info(f"{len(finished_jobs):d} of {len(job_df.index):d} Jobs have already finished and will be skipped.")
        job_df = job_df.drop(index=finished_jobs)
    return job_df


def _run(job_df, cwd, output_dir, flavour, num_processes, run_local, additional_htc_parameters):
    if run_local:
        LOG.info(f"Running {len(job_df.index)} jobs locally in {num_processes:d} processes.")
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(_execute_shell, job_df.iterrows())

    else:
        LOG.info(f"Submitting {len(job_df.index)} jobs on htcondor, flavour '{flavour}'.")
        # create submission file
        subfile = htcutils.make_subfile(cwd, job_df, output_dir=output_dir, duration=flavour,
                                        **additional_htc_parameters)
        # submit to htcondor
        htcutils.submit_jobfile(subfile)


# Sub Functions ----------------------------------------------------------------


def _setup_folders(job_df, working_directory):
    LOG.debug("Setting up folders: ")
    job_df[COLUMN_JOB_DIRECTORY] = list(map(_return_job_dir, zip([working_directory] * len(job_df), job_df.index)))
    for job_dir in job_df[COLUMN_JOB_DIRECTORY]:
        try:
            os.mkdir(job_dir)
        except IOError:
            LOG.debug(f"   created '{job_dir}'.")
        else:
            LOG.debug(f"   failed '{job_dir}' (might already exist).")
    return job_df


def _job_was_successful(job_row, output_dir, files):
    output_dir = os.path.join(job_row[COLUMN_JOB_DIRECTORY], output_dir)
    success = os.path.isdir(output_dir)
    if success and files is not None and len(files):
        for f in files:
            success &= os.path.isfile(os.path.join(output_dir, f))
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


def _set_auto_tfs_column_types(df):
    return df.apply(partial(pd.to_numeric, errors='ignore'))


def _find_named_variables_in_mask(mask):
    return set(re.findall(r"%\((\w+)\)", mask))


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
