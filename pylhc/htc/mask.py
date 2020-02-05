"""
MASK Resolver
------------------

Package to resolve script masks.

:module: htc.mask
:author: mihofer

"""
import logging
from pathlib import Path

from pandas import DataFrame

from pylhc.htc.utils import COLUMN_JOB_DIRECTORY, COLUMN_JOB_FILE

LOG = logging.getLogger(__name__)


def create_madx_jobs_from_mask(job_df: DataFrame, maskfile: Path, replace_keys: dict, file_ext: str = 'madx'):
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

    with maskfile.open('r') as mfile:
        template = mfile.read()

    jobname = maskfile.with_suffix('').name
    jobs = [None] * len(job_df)
    for idx, (jobid, values) in enumerate(job_df.iterrows()):
        jobfile_fullpath = (Path(values[COLUMN_JOB_DIRECTORY]) / jobname).with_suffix(f'.{file_ext}')

        with jobfile_fullpath.open('w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobfile_fullpath.name
    job_df[COLUMN_JOB_FILE] = jobs
    return job_df


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
