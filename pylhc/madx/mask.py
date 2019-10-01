"""
MADX MASK Resolver
------------------

Takes path to mask file, list of parameter to be replaced and pandas dataframe containg per job 
the job directory where processed mask is to be put, and columns containg the parameter values with column named like replace parameterself.
Job directories have to be created beforehand. Processed madx mask has the same filename as mask but with file extension .madx.
Input Dataframe is returned with additonal column containg path to the processed madx files.

"""
import re
import os
from htc.utils import COLUMN_SHELL_SCRIPTS, COLUMN_JOB_DIRECTORY, COLUMN_JOBS


def create_madx_jobs_from_mask(job_df, maskfile, replace_keys):

    with open(maskfile, 'r') as mfile:
        template = mfile.read()

    jobname = os.path.splitext(os.path.basename(maskfile))[0]
    jobs = [None] * len(job_df)
    for idx, values in job_df.iterrows():
        jobdir = os.path.join(values[COLUMN_JOB_DIRECTORY], f'{jobname}.madx')
        with open(jobdir, 'w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobdir
    job_df[COLUMN_JOBS] = jobs
    return job_df


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
