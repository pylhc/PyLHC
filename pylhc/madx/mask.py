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
MASK_ENDING = '.mask'


def create_madx_from_mask(maskfile, replace_keys, job_df):

    with open(maskfile, 'r') as mfile:
        template = mfile.read()

    jobname = re.sub(MASK_ENDING, '', maskfile)
    jobs = [None] * len(job_df)
    for idx, values in job_df.iterrows():
        jobdir = os.path.join(values['Job_directory'], f'{jobname}.madx')
        with open(jobdir, 'w') as madxjob:
            madxjob.write(template % dict(zip(replace_keys, values[list(replace_keys)])))
        jobs[idx] = jobdir
    return jobs


if __name__ == '__main__':
    raise EnvironmentError(f"{__file__} is not supposed to run as main.")
