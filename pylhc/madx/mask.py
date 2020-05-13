"""
MADX MASK Resolver
------------------

Takes path to mask file, list of parameter to be replaced and pandas dataframe containg per job 
the job kick_directory where processed mask is to be put, and columns containg the parameter values with column named like replace parameterself.
Job directories have to be created beforehand. Processed madx mask has the same filename as mask but with file extension .madx.
Input Dataframe is returned with additonal column containg path to the processed madx files.

:module: madx.mask
:author: mihofer

"""
import logging
import os

from pylhc.htc.utils import COLUMN_JOB_DIRECTORY, COLUMN_JOB_FILE

LOG = logging.getLogger(__name__)


def create_madx_jobs_from_mask(job_df, maskfile, replace_keys):

    with open(maskfile, 'r') as mfile:
        template = mfile.read()

    jobname = os.path.splitext(os.path.basename(maskfile))[0]
    jobs = [None] * len(job_df)
    for idx, (jobid, values) in enumerate(job_df.iterrows()):
        jobfile_name = f'{jobname}.madx'
        jobfile_fullpath = os.path.join(values[COLUMN_JOB_DIRECTORY], jobfile_name)

        with open(jobfile_fullpath, 'w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobfile_name
    job_df[COLUMN_JOB_FILE] = jobs
    return job_df


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
