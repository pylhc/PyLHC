import subprocess
import os
import htcondor

SHEBANG = "#!/bin/bash"
SUBFILE = "queuehtc.sub"

MADX_PATH = '/afs/cern.ch/user/m/mad/bin/madx'
PYTHON_PATH = '/afs/cern.ch/eng/sl/lintrack/anaconda3/bin/python'
SAD_PATH = 'Road to nowhere'

EXECUTEABLEPATH = {'madx': MADX_PATH,
                   'python': PYTHON_PATH,
                   'sad': SAD_PATH}

CMD_SUBMIT = "condor_submit"

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


def create_multijob_for_bashfiles(folder, n_files, mask, duration="workday"):
    """ Function to create a HTCondor job assuming n_files bash-files. """
    dura_key, dura_val = _get_duration(duration)

    job = htcondor.Submit({
        "MyId": "htcondor",
        "universe": "vanilla",
        "executable": os.path.join(folder, f'{mask}.$(Process).sh'),
        "arguments": "$(ClusterId) $(ProcId)",
        "initialdir": os.path.join(folder),
        "transfer_output_files": "Data",
        "notification": "error",
        "output": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).out"),
        "error": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).err"),
        "log": os.path.join("$(initialdir)", "$(MyId).$(ClusterId).$(ProcId).log"),
        dura_key: dura_val,
    })
    job.setQArgs(f"{n_files}")
    return job


def make_subfile(folder, n_files, mask, duration):
    job = create_multijob_for_bashfiles(folder, n_files, mask, duration)
    return create_subfile_from_job(folder, job)

# For bash #####################################################################


def write_bash(job_files, jobtype='madx'):
    shell_scripts=[]
    for idx, job in enumerate(job_files):
        jobfile = f'{job}.sh'
        with open(jobfile, 'w') as f:

            f.write(SHEBANG + "\n")
            f.write('mkdir Data\n')
            f.write(f'{EXECUTEABLEPATH[jobtype]} {job}  \n')
        shell_scripts.append(jobfile)
    return shell_scripts


# Helper #######################################################################


def _get_duration(duration):
    if duration in JOBFLAVOURS:
        return "+JobFlavour", 'f"{duration}"'
    if isinstance(duration, int):  # for the moment this option is excluded in entrypoint
        return "+MaxRuntime", f'duration'
    else:
        raise TypeError(
            f"Duration is not given in correct format, provide either seconds as int or str from list {JOBFLAVOURS}")

# Script Mode ##################################################################


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
