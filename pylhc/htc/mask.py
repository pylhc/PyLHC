"""
Mask Resolver
-------------

This module provides functionality to resolve and write script masks for ``HTCondor`` jobs
submission.
"""
import logging
import re
from pathlib import Path

import pandas as pd

from pylhc.htc.utils import COLUMN_JOB_DIRECTORY, COLUMN_JOB_FILE

LOG = logging.getLogger(__name__)


def create_jobs_from_mask(
    job_df: pd.DataFrame, maskfile: Path, replace_keys: dict, file_ext: str
) -> pd.DataFrame:
    """
    Takes path to mask file, list of parameter to be replaced and pandas dataframe containg per job
    the job directory where processed mask is to be put, and columns containing the parameter values
    with column named like replace parameters. Job directories have to be created beforehand.
    Processed (madx) mask has the same filename as mask but with the given file extension.
    Input Dataframe is returned with additional column containing path to the processed script
    files.

    Args:
        job_df (pd.DataFrame): Job parameters as defined in description.
        maskfile: `Path` object to the mask file.
        replace_keys: keys to be replaced (must correspond to columns in ``job_df``).
        file_ext: file extention to use (defaults to **madx**).

    Returns:
        The provided ``job_df`` but with added path to the scripts.
    """
    with maskfile.open("r") as mfile:
        template = mfile.read()

    jobname = maskfile.with_suffix("").name
    jobs = [None] * len(job_df)
    for idx, (jobid, values) in enumerate(job_df.iterrows()):
        jobfile_fullpath = (Path(values[COLUMN_JOB_DIRECTORY]) / jobname).with_suffix(file_ext)

        with jobfile_fullpath.open("w") as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobfile_fullpath.name
    job_df[COLUMN_JOB_FILE] = jobs
    return job_df


def find_named_variables_in_mask(mask):
    return set(re.findall(r"%\((\w+)\)", mask))


def generate_jobdf_index(old_df, jobid_mask, keys, values):
    """ Generates index for jobdf from mask for job_id naming. """
    if not jobid_mask:
        nold = len(old_df.index) if old_df is not None else 0
        start = nold-1 if nold > 0 else 0
        return range(start, start + values.shape[0])
    return [jobid_mask % dict(zip(keys, v)) for v in values]


if __name__ == '__main__':
if __name__ == "__main__":
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
