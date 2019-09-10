import subprocess
import os
try:
    import htcondor

MADX_PATH = '/afs/cern.ch/user/m/mad/bin/madx'
PYTHON_PATH = '/afs/cern.ch/eng/sl/lintrack/anaconda3/bin/python'
SAD_PATH = ''


def write_condor_script(directory, jobname, njobs=1, flavour='workday'):

    with open(directory + '/htjob.sub', 'w') as file:
        file.write('Universe        = vanilla\n')
        file.write(f'Executable      = {jobname}.$(Process).sh\n')
        file.write('getenv          = True\n')

        file.write('notification    = error\n')

        file.write('\ntransfer_output_files   = Data\n')

        file.write('\nerror   = Output/err.$(process).txt\n')
        file.write('output  = Output/out.$(process).txt\n')
        file.write('log     = Output/$(Cluster).log\n')

        file.write(f'\n+JobFlavour = "{flavour}"\n')

        file.write(f'\ninitialdir    = {directory}\n')
        file.write(f'queue {njobs}\n')


def launch_condor_script(directory):
    os.chdir(directory)
    subprocess.call('condor_submit ./htjob.sub', shell=True)


def create_and_launch_condor_script(directory, jobname, njobs=1, flavour='workday'):
    write_condor_script(directory, jobname, njobs, flavour)
    launch_condor_script(directory)


def create_executeables(directory, jobname, njobs, mask, jobtype='madx'):

    executeablepath = {'madx': MADX_PATH,
                       'python': PYTHON_PATH,
                       'sad': SAD_PATH}
    for jobid in range(njobs):
        jobfile = os.path.join(directory, f'{jobname}.{jobid}.sh')
        with open(jobfile, 'w') as file:

            file.write('#!/bin/bash\n')
            file.write('\nmkdir Data\n')
            file.write(f'\n{executeablepath[jobtype]} {mask}.{jobid}  \n')
