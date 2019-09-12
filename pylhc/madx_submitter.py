import os
import re
import tfs
import itertools
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entrypoint_parser import save_options_to_config
from generic_parser.entry_datatypes import DictAsString
import htc.utils
from htc.utils import JOBFLAVOURS, JOBDIRECTORY_NAME, HTCONDOR_JOBLIMIT, OUTPUT_DIR
from madx import mask
from madx.mask import MASK_ENDING

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
    opt = check_opts(opt)

    save_options_to_config(os.path.join(opt.working_directory, 'config.ini'), opt)

    values_grid = list(itertools.product(*opt.replace_dict.values()))
    njobs = len(values_grid)

    if njobs > HTCONDOR_JOBLIMIT:
        print('Submitting too many jobs')
        exit()
    setup_folders(njobs, opt.working_directory)

    job_df = pd.DataFrame({'JobId': range(njobs)})
    job_df = job_df.join(pd.DataFrame(columns=list(opt.replace_dict.keys()),
                                      data=values_grid))
    # creating all madx jobs
    job_df['Jobs'] = mask.create_madx_from_mask(opt.working_directory,
                                                opt.mask,
                                                opt.replace_dict.keys(),
                                                values_grid
                                                )
    # creating all shell scripts
    job_df['Shell_script'] = htc.utils.write_bash(opt.working_directory, job_df['Jobs'], 'madx')
    job_df['Job_directory'] = list(map(os.path.dirname, job_df['Shell_script']))

    tfs.write(os.path.join(opt.working_directory, JOBSUMMARY_FILE), job_df)

    if opt.resume_jobs:
        job_df = tfs.read(os.path.join(opt.working_directory, JOBSUMMARY_FILE))
        unfinished_jobs = [idx for idx, row in job_df.iterrows() if os.path.exists(os.path.join(row['Job_directory'], OUTPUT_DIR))]
        job_df = job_df.drop(index=unfinished_jobs)

    if opt.run_local:
        pool = multiprocessing.Pool(processes=opt.num_processes)
        pool.map(execute_shell, job_df.iterrows())

    else:
        # create submission file
        subfile = htc.utils.make_subfile(opt.working_directory,
                                         job_df,
                                         opt.jobflavour)
        # submit to htcondor
        htc.utils.submit_jobfile(subfile)


def setup_folders(njobs, working_directory):
    os.chdir(working_directory)
    for idx in range(njobs):
        try:
            os.mkdir(f'{JOBDIRECTORY_NAME}.{idx}')
        except:
            pass


def execute_shell(df_row):
    idx, column = df_row
    with open(os.path.join(column['Job_directory'], 'log.tmp'), 'w') as logfile:
        process = subprocess.Popen(['sh', column['Shell_script']],
                                   shell=False,
                                   stdout=logfile,
                                   stderr=subprocess.STDOUT,
                                   cwd=column['Job_directory'])

    status = process.wait()
    return status


def check_opts(opt):

    with open(os.path.join(opt.working_directory, opt.mask)) as inputmask:  # checks that mask and dir are there

        mask = inputmask.read()
        keys_not_found = [k for k in opt.replace_dict.keys() if f'%({k})s' not in mask]
        # log.info (keys not found in mask)
        [opt.replace_dict.pop(key) for key in keys_not_found]  # removes all keys which are not present in mask
        if opt.replace_dict == {}:
            raise AttributeError('Empty replacedictionary')

    return opt


if __name__ == '__main__':
    # main(
    #     mask='jobB1inj.2negCorr.BBeat_full.mask',
    #     working_directory='/afs/cern.ch/work/m/mihofer2/public/MDs/MD3603/Simulations/ForcedDA',
    #     jobflavour='workday',
    #     run_local=True,
    #     resume_jobs=False,
    #     replace_dict={"AMPLITUDEX": np.linspace(0.0, 0.004, 41),
    #                   "AMPLITUDEY": np.linspace(0.0, 0.004, 41)}
    #     )

    main(
        mask='jobB1inj.2negCorr.BBeat.mask',
        working_directory='/afs/cern.ch/work/m/mihofer2/public/MDs/MD3603/Simulations/ForcedDA',
        jobflavour='workday',
        run_local=True,
        resume_jobs=False,
        replace_dict={"SEEDRAN": [0, 5],
                      }
        )
