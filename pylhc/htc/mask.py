"""
MASK Resolver
------------------

Package to resolve script masks.

:module: madx.mask
:author: mihofer

"""
import logging
import os

from pylhc.htc.utils import COLUMN_JOB_DIRECTORY, COLUMN_JOB_FILE

LOG = logging.getLogger(__name__)


def create_madx_jobs_from_mask(job_df, maskfile, replace_keys, file_ext='madx'):
    """
    Takes path to mask file, list of parameter to be replaced and pandas dataframe containg per job
    the job directory where processed mask is to be put, and columns containing the parameter values
    with column named like replace parameters. Job directories have to be created beforehand.
    Processed madx mask has the same filename as mask but with the given file extension.
    Input Dataframe is returned with additional column containing path to the processed script files.


    Args:
        job_df: Job parameters as defined in description
        maskfile: path to mask file
        replace_keys: keys to be replaced (must correspond to columns in job_df)
        file_ext: file extention to use (defaults to 'madx')

    Returns: job_df but with added path to the scripts

    """

    with open(maskfile, 'r') as mfile:
        template = mfile.read()

    jobname = os.path.splitext(os.path.basename(maskfile))[0]
    jobs = [None] * len(job_df)
    for idx, (jobid, values) in enumerate(job_df.iterrows()):
        jobfile_name = f'{jobname}.{file_ext}'
        jobfile_fullpath = os.path.join(values[COLUMN_JOB_DIRECTORY], jobfile_name)

        with open(jobfile_fullpath, 'w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobfile_name
    job_df[COLUMN_JOB_FILE] = jobs
    return job_df


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
