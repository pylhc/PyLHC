import re
import os
MASK_ENDING = '.mask'


def create_madx_from_mask(cwd, maskfile, replace_keys, replace_values):

    with open(os.path.join(cwd, maskfile), 'r') as mfile:
        template = mfile.read()

    jobname = re.sub(MASK_ENDING, '', maskfile)
    jobs = [None] * len(replace_values)
    for idx, values in enumerate(replace_values):
        jobdir = os.path.join(cwd, f'Job.{idx}', f'{jobname}.madx')
        with open(jobdir, 'w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values)))
        jobs[idx] = jobdir
    return jobs
