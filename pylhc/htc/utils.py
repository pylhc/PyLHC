"""
HTC Utils
----------

Functions allowing to create HTCondor jobs and submit them.

write_bash creates bash scripts executing either a python or madx script. 
Takes dataframe, job type, and optional additional cmd line arguments for script.
A shell script is created in each job directory in the dataframe.

make_subfile takes the job dataframe and creates the .sub required for submissions to HTCondor.
The .sub file will be put in the working directory. 
The maximum runtime of one job can be specified, standard is 8h.

"""
import subprocess
import os
import htcondor

SHEBANG = "#!/bin/bash"
SUBFILE = "queuehtc.sub"
BASH_FILENAME = 'Job'

HTCONDOR_JOBLIMIT = 100000

EXECUTEABLEPATH = {'madx': '/afs/cern.ch/user/m/mad/bin/madx',
                   'python3': '/afs/cern.ch/eng/sl/lintrack/anaconda3/bin/python',
                   'python2': '/afs/cern.ch/eng/sl/lintrack/miniconda2/bin/python',
                   }

CMD_SUBMIT = "condor_submit"
JOBFLAVOURS = ('espresso',  # 20 min
               'microcentury',  # 1 h
               'longlunch',  # 2 h
               'workday',  # 8 h
               'tomorrow',  # 1 d
               'testmatch',  # 3 d
               'nextweek'  # 1 w
               )


COLUMN_SHELL_SCRIPT = 'ShellScript'
COLUMN_JOB_DIRECTORY = 'JobDirectory'
COLUMN_JOB_FILE = "JobFile"


# Subprocess Methods ###########################################################


def create_subfile_from_job(cwd, job):
    """ Write file to submit to htcondor """
    subfile = os.path.join(cwd, SUBFILE)

    with open(subfile, "w") as f:
        f.write(str(job))
    return subfile


def submit_jobfile(jobfile):
    """ Submit subfile to htcondor via subprocess """
    _start_subprocess([CMD_SUBMIT, jobfile])


def _start_subprocess(command):
    process = subprocess.Popen(command, shell=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,)
    status = process.wait()
    return status


# Job Creation #################################################################


def create_multijob_for_bashfiles(job_df, output_dir, duration="workday"):
    """ Function to create a HTCondor job assuming n_files bash-files. """
    dura_key, dura_val = _get_duration(duration)

    job = htcondor.Submit({
        "MyId": "htcondor",
        "universe": "vanilla",
        "arguments": "$(ClusterId) $(ProcId)",
        "transfer_output_files": output_dir,
        "notification": "error",
        "output": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).out"),
        "error": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).err"),
        "log": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).log"),
        "on_exit_remove": '(ExitBySignal == False) && (ExitCode == 0)',
        "max_retries": '3',
        "requirements": 'Machine =!= LastRemoteHost',
        dura_key: dura_val,
    })
    # ugly but job.setQArgs doesn't take string containing \n
    scripts = [os.path.join(*parts) for parts in zip(job_df[COLUMN_JOB_DIRECTORY], job_df[COLUMN_SHELL_SCRIPT])]
    args = "\n".join([",".join(parts) for parts in zip(scripts, job_df[COLUMN_JOB_DIRECTORY])])
    queueArg = f"queue executable, initialdir from (\n{args})"
    job = str(job) + queueArg
    return job

# Main functions ###############################################################


def make_subfile(cwd, job_df, output_dir,  duration):
    job = create_multijob_for_bashfiles(job_df, output_dir, duration)
    return create_subfile_from_job(cwd, job)


def write_bash(job_df, output_dir, jobtype='madx', cmdline_arguments={}):
    if len(job_df.index) > HTCONDOR_JOBLIMIT:
        raise AttributeError('Submitting too many jobs for HTCONDOR')

    shell_scripts = [None] * len(job_df.index)
    for idx, (jobid, job) in enumerate(job_df.iterrows()):
        bash_file = f'{BASH_FILENAME}.{jobid}.sh'
        jobfile = os.path.join(job[COLUMN_JOB_DIRECTORY], bash_file)
        with open(jobfile, 'w') as f:
            f.write(f"{SHEBANG}\n")
            f.write(f'mkdir {output_dir}\n')
            cmds = ' '.join([f'--{param} {val}' for param, val in cmdline_arguments.items()])
            f.write(
                f'{EXECUTEABLEPATH[jobtype]} {os.path.join(job[COLUMN_JOB_DIRECTORY], job[COLUMN_JOB_FILE])} {cmds}\n'
            )
        shell_scripts[idx] = bash_file
    job_df[COLUMN_SHELL_SCRIPT] = shell_scripts
    return job_df


# Helper #######################################################################


def _get_duration(duration):
    if duration in JOBFLAVOURS:
        return "+JobFlavour", f'"{duration}"'
    else:
        raise TypeError(
            f"Duration is not given in correct format, provide str from list {JOBFLAVOURS}")

# Script Mode ##################################################################


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
