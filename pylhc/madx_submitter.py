import os
import re
import itertools
import subprocess
import multiprocessing
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entry_datatypes import DictAsString
import htc.utils
from htc.utils import JOBFLAVOURS
from madx import mask
from madx.mask import MASK_ENDING


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
        flags="--resume",
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
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    opt = check_opts(opt)

    values_grid = list(itertools.product(*opt.replace_dict.values()))
    njobs = len(values_grid)

    if njobs > 10000:
        print('Submitting too many jobs')
        exit()

    # creating all madx jobs
    jobs = mask.create_madx_from_mask(
                                      os.path.join(opt.working_directory, opt.mask),
                                      opt.replace_dict.keys(),
                                      values_grid
                                     )
    # creating all shell scripts
    shell_scripts = htc.utils.write_bash(jobs, 'madx')

    if opt.run_local:
        pool = multiprocessing.Pool(processes=4)
        pool.map(execute_shells, shell_scripts)
    else:
        # create submission file 
        subfile = htc.utils.make_subfile(opt.working_directory,
                                         njobs,
                                         re.sub(MASK_ENDING, '', opt.mask),
                                         opt.jobflavour)
        # submit to htcondor
        htc.utils.submit_jobfile(subfile)


def execute_shells(shell_script):
    os.chmod(shell_script, 477)
    job_dir = re.sub('.sh', '', shell_script)
    try:
        os.mkdir(f'{job_dir}_dir')
    except:
        pass
    os.chdir(f'{job_dir}_dir')
    process = subprocess.Popen(shell_script, shell=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    status = process.wait()
    return status


def check_opts(opt):

    with open(os.path.join(opt.working_directory, opt.mask)) as inputmask:  # checks that mask and dir are there
        keys_not_found = []
        mask = inputmask.read()
        for key in opt.replace_dict.keys():
            if key not in mask:
                keys_not_found.append(key)
        # log.info (keys not found in mask)
        [opt.replace_dict.pop(key) for key in keys_not_found]  # removes all keys whihc are not present in mask
        if opt.replace_dict == {}:
            raise AttributeError('Empty replacedictionary')

    return opt


if __name__ == '__main__':
    main(
        mask='jobB1inj.2negCorr.BBeat.mask',
        working_directory='/afs/cern.ch/work/m/mihofer2/public/MDs/MD3603/Simulations/ForcedDA',
        jobflavour='workday',
        run_local=True,
        replace_dict={"%SEEDRAN": [0, 1, 2, 3]}
        )
