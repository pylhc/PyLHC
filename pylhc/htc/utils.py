import subprocess
import os
import htcondor

SHEBANG = "#!/bin/bash"
SUBFILE = "queuehtc.sub"
BASH_FILENAME = 'Job'
JOBDIRECTORY_NAME = 'Job'
HTCONDOR_JOBLIMIT = 100000

EXECUTEABLEPATH = {'madx': '/afs/cern.ch/user/m/mad/bin/madx',
                   'python': '/afs/cern.ch/eng/sl/lintrack/anaconda3/bin/python',
                   }
CMD_SUBMIT = "condor_submit"
OUTPUT_DIR = 'Outputdata'
JOBFLAVOURS = ('espresso',  # 20 min
               'microcentury',  # 1 h
               'longlunch',  # 2 h
               'workday',  # 8 h
               'tomorrow',  # 1 d
               'testmatch',  # 3 d
               'nextweek'  # 1 w
               )


# Subprocess Methods ###########################################################


def create_subfile_from_job(folder, job):
    """ Write file to submit to htcondor """
    subfile = os.path.join(folder, SUBFILE)

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


def create_multijob_for_bashfiles(folder, job_df, duration="workday"):
    """ Function to create a HTCondor job assuming n_files bash-files. """
    dura_key, dura_val = _get_duration(duration)

    job = htcondor.Submit({
        "MyId": "htcondor",
        "universe": "vanilla",
        "arguments": "$(ClusterId) $(ProcId)",
        "transfer_output_files": OUTPUT_DIR,
        "notification": "error",
        "output": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).out"),
        "error": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).err"),
        "log": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).log"),
        "on_exit_remove": '(ExitBySignal == False) && (ExitCode == 0)',
        "max_retries": '3',
        "requirements": 'Machine =!= LastRemoteHost',
        dura_key: dura_val,
    })

    queueArg = f"queue executable, initialdir from (\n{job_df.to_csv(index=False, header=False, columns=['Shell_script', 'Job_directory'])})"
    job = str(job) + queueArg

    return job


def make_subfile(folder, n_files, duration):
    job = create_multijob_for_bashfiles(folder, n_files, duration)
    return create_subfile_from_job(folder, job)

# For bash #####################################################################


def write_bash(cwd, job_files, jobtype='madx', cmdline_arguments={}):
    shell_scripts = []
    if len(job_files) > HTCONDOR_JOBLIMIT:
        raise AttributeError('Submitting too many jobs for HTCONDOR')
    for idx, job in enumerate(job_files):
        jobfile = os.path.join(cwd, f'{JOBDIRECTORY_NAME}.{idx}', f'{BASH_FILENAME}.{idx}.sh')
        with open(jobfile, 'w') as f:

            f.write(SHEBANG + "\n")
            f.write(f'mkdir {OUTPUT_DIR}\n')
            cmds = ' '.join([f'--{param} {val}' for param, val in cmdline_arguments.items()])
            f.write(f'{EXECUTEABLEPATH[jobtype]} {job} {cmds}\n')
        shell_scripts.append(jobfile)
    return shell_scripts


# Helper #######################################################################


def _get_duration(duration):
    if duration in JOBFLAVOURS:
        return "+JobFlavour", f"{duration}"
    else:
        raise TypeError(
            f"Duration is not given in correct format, provide str from list {JOBFLAVOURS}")

# Script Mode ##################################################################


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
