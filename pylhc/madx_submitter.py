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


"""
import itertools
import multiprocessing
import os
import subprocess

import numpy as np
import pandas as pd
import tfs
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import save_options_to_config

import htc.utils
import madx
from htc.utils import JOBFLAVOURS, JOBDIRECTORY_NAME, HTCONDOR_JOBLIMIT, OUTPUT_DIR

JOBSUMMARY_FILE = 'Jobs.tfs'


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        flags="--mask",
        name="mask",
        type=str,
        required=True,
        help="Madx mask to use",
    )
    params.add_parameter(
        flags="--working_directory",
        name="working_directory",
        type=str,
        required=True,
        help="Directory where data should be put",
    )
    params.add_parameter(
        flags="--jobflavour",
        name="jobflavour",
        type=str,
        choices=JOBFLAVOURS,
        default='workday',
        help="Jobflavour to give rough estimate of runtime of one job ",
    )
    params.add_parameter(
        flags="--local",
        name="run_local",
        action="store_true",
        help="Flag to run the jobs on the local machine. Not suggested.",
    )
    params.add_parameter(
        flags="--resume_jobs",
        name="resume_jobs",
        action="store_true",
        help="Only do jobs that did not work.",
    )
    params.add_parameter(
        flags="--append_jobs",
        name="append_jobs",
        action="store_true",
        help="Flag to rerun job with finer/wider grid, already existing points will not be reexecuted.",
    )
    params.add_parameter(
        flags="--replace_dict",
        name="replace_dict",
        help="Dict containing the str to replace as keys and values a list of parameters to replace",
        type=DictAsString,
        required=True
    )
    params.add_parameter(
        flags="--num_processes",
        name="num_processes",
        help="number of processes to be used if run locally",
        type=int,
        default=4
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    opt = _check_opts(opt)
    save_options_to_config(os.path.join(opt.working_directory, 'config.ini'), opt)

    job_df = _create_jobs(opt.working_directory, opt.mask, opt.replace_dict, opt.append_jobs)
    job_df = _drop_already_run_jobs(job_df, opt.working_directory, opt.resume_jobs or opt.append_jobs)
    _run(job_df, opt.working_directory, opt.jobflavour, opt.num_processes, opt.run_local)


# Main Functions ---------------------------------------------------------------


def _create_jobs(cwd, mask, replace_dict, append_jobs):
    values_grid = np.array(list(itertools.product(*replace_dict.values())))

    if append_jobs:
        job_df = tfs.read(os.path.join(cwd, JOBSUMMARY_FILE), index='JobId')
        mask = [elem not in job_df[replace_dict.keys()].values for elem in values_grid]
        njobs = mask.count(True)
        values_grid = values_grid[mask]

    else:
        njobs = len(values_grid)
        job_df = pd.DataFrame()

    if njobs > HTCONDOR_JOBLIMIT:
        print('Submitting too many jobs')
        exit()

    job_df = job_df.append(pd.DataFrame(columns=list(replace_dict.keys()),
                                        data=values_grid), ignore_index=True, sort=False)

    job_df = _setup_folders(job_df, cwd)

    # creating all madx jobs
    job_df = madx.mask.create_madx_jobs_from_mask(job_df, mask, replace_dict.keys())

    # creating all shell scripts
    job_df = htc.utils.write_bash(job_df, 'madx')

    tfs.write(os.path.join(cwd, JOBSUMMARY_FILE), job_df, save_index='JobId')
    return job_df


def _drop_already_run_jobs(job_df, cwd, extend_jobs):
    if extend_jobs:
        job_df = tfs.read(os.path.join(cwd, JOBSUMMARY_FILE))
        unfinished_jobs = [idx for idx, row in job_df.iterrows() if os.path.isdir(os.path.join(row['Job_directory'], OUTPUT_DIR))]
        job_df = job_df.drop(index=unfinished_jobs)
    return job_df


def _run(job_df, cwd, flavour, num_processes, run_local):
    if run_local:
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(_execute_shell, job_df.iterrows())

    else:
        # create submission file
        subfile = htc.utils.make_subfile(cwd, job_df, flavour)
        # submit to htcondor
        htc.utils.submit_jobfile(subfile)


# Sub Functions ----------------------------------------------------------------


def _setup_folders(job_df, working_directory):
    job_df['Job_directory'] = list(map(_return_job_dir, zip([working_directory] * len(job_df), job_df.index)))
    for job_dir in job_df['Job_directory']:
        try:
            os.mkdir(job_dir)
        except IOError:
            pass
    return job_df


def _return_job_dir(inp):
    return os.path.join(inp[0], f'{JOBDIRECTORY_NAME}.{inp[1]}')


def _execute_shell(df_row):
    idx, column = df_row
    with open(os.path.join(column['Job_directory'], 'log.tmp'), 'w') as logfile:
        process = subprocess.Popen(['sh', column['Shell_script']],
                                   shell=False,
                                   stdout=logfile,
                                   stderr=subprocess.STDOUT,
                                   cwd=column['Job_directory'])

    status = process.wait()
    return status


def _check_opts(opt):

    if opt.resume_jobs and opt.append_jobs:
        raise AttributeError('Select either Resume jobs or Append jobs')

    with open(opt.mask, "r") as inputmask:  # checks that mask and dir are there

        mask = inputmask.read()
        keys_not_found = [k for k in opt.replace_dict.keys() if f'%({k})s' not in mask]
        # log.info (keys not found in mask)
        [opt.replace_dict.pop(key) for key in keys_not_found]  # removes all keys which are not present in mask
        if opt.replace_dict == {}:
            raise AttributeError('Empty replacedictionary')

    return opt


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
