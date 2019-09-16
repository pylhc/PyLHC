import re
import os
MASK_ENDING = '.mask'


def create_madx_from_mask(cwd, maskfile, replace_keys, job_df):

    with open(os.path.join(cwd, maskfile), 'r') as mfile:
        template = mfile.read()

    jobname = re.sub(MASK_ENDING, '', maskfile)
    jobs = [None] * len(job_df)
    for idx, values in job_df.iterrows():
        jobdir = os.path.join(values['Job_directory'], f'{jobname}.madx')
        with open(jobdir, 'w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobdir
    return jobs
