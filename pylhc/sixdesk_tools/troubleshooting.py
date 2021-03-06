"""
SixDesk Troubleshooting tools
-----------------------------

Some useful functions to troubleshoot the SixDesk output.
"""
from pathlib import Path

from omc3.utils import logging_tools

from pylhc.autosix import get_jobs_and_values
from pylhc.constants.autosix import get_stagefile_path, STAGES, get_track_path, get_workspace_path, \
    SIXTRACK_INPUT_CHECK_FILES, SIXTRACK_OUTPUT_FILES

LOG = logging_tools.get_logger(__name__)


# Stages -----------------------------------------------------------------------

def get_last_stage(jobname, basedir):
    """ Get the last run stage of job `jobname`. """
    stage_file = get_stagefile_path(jobname, basedir)
    last_stage = stage_file.read_text().strip('\n').split('\n')[-1]
    return last_stage


# Set Stages ---

def set_stages_for_setup(basedir, stage_name, jobid_mask, replace_dict):
    """ Sets the last run stage for all jobs from given job-setups. """
    jobs, _ = get_jobs_and_values(jobid_mask, **replace_dict)
    set_stages_for_jobs(basedir, jobs, stage_name)


def set_stages_for_all_jobs(basedir, stage_name):
    """ Sets the last run stage for all jobs in sexdeskbase dir. """
    jobs = get_all_jobs_in_base(basedir)
    set_stages_for_jobs(basedir, jobs, stage_name)


def set_stages_for_jobs(basedir, jobs, stage_name):
    """ Sets the last run stage of all given jobs to `stage_name`. """
    if stage_name not in STAGES:
        raise ValueError(f"Unknown stage '{stage_name}'")

    for jobname in jobs:
        stage_file = get_stagefile_path(jobname, basedir)
        if stage_file.exists():
            stage_file.unlink()
        for stage in STAGES:
            with open(stage_file, "a+") as f:
                f.write(f"{stage}\n")

            if stage == stage_name:
                break


# Check Stages ---

def check_stages_for_setup(basedir, stage_name, jobid_mask, replace_dict):
    """ Check the last run stage for all jobs from given job-setups. """
    jobs, _ = get_jobs_and_values(jobid_mask, **replace_dict)
    check_stages_for_jobs(basedir, jobs)


def check_stages_for_all_jobs(basedir):
    """ Checks the last run stages for all Jobs in sixdeskbase dir. """
    jobs = get_all_jobs_in_base(basedir)
    check_stages_for_jobs(basedir, jobs)


def check_stages_for_jobs(basedir, jobs):
    """ Logs names of all last run stages for given jobs. """
    for jobname in jobs:
        last_stage = get_last_stage(jobname, basedir)
        LOG.info(f"'{jobname}' at stage '{last_stage}'")


# Complete check for failure ---------------------------------------------------

def find_obviously_failed_sixtrack_submissions(basedir: Path):
    """ Checks in jobs in `track` whether the directory structure seems to be created
    and if there is output data. This checks only the first directories found,
    to speed up this process. For a more precise scan see check_sixtrack_output_data.
    """
    jobs = []
    for job in get_all_jobs_in_base(basedir):
        try:
            LOG.debug(str(job))
            track = get_track_path(jobname=job, basedir=base)
            first_seed = get_first_dir(track) / 'simul'
            tunes = get_first_dir(first_seed)
            amp = get_first_dir(tunes, '*_*')
            turns = get_first_dir(amp, 'e*')
            angle = get_first_dir(turns, ".*")

            file_names = [f.name for f in angle.glob("*")]
            out_files_present = [f for f in SIXTRACK_OUTPUT_FILES if f in file_names]
            if not len(out_files_present):
                raise OSError(str(get_workspace_path(jobname=job, basedir=base)))
        except OSError as e:
            LOG.error(f"{e.args[0]} (stage: {get_last_stage(jobname=job, basedir=basedir)})")
            jobs.append(str(job))
    return jobs


def check_sixtrack_output_data(jobname: str, basedir: Path):
    """ Presence checks for SixDesk tracking output data.

    This checks recursively all directories in `track`.
    Will be busy for a while.
    """
    track_path = get_track_path(jobname, basedir)
    seed_dirs = list(track_path.glob("[0-9]")) + list(track_path.glob("[0-9][0-9]"))
    if not len(seed_dirs):
        raise OSError(f"No seed-dirs present in {str(track_path)}.")

    for seed_dir in seed_dirs:
        if not seed_dir.is_dir():
            continue

        simul_path = seed_dir / 'simul'
        tunes_dirs = list(simul_path.glob("*"))
        if not len(tunes_dirs):
            raise OSError(f"No tunes-dirs present in {str(seed_dir)}.")

        for tunes_dir in tunes_dirs:
            if not tunes_dir.is_dir():
                continue

            amp_dirs = list(tunes_dir.glob("*_*"))
            if not len(amp_dirs):
                raise OSError(f"No amplitude-dirs present in {str(tunes_dir)}.")

            for amp_dir in amp_dirs:
                if not amp_dir.is_dir():
                    continue

                turns_dirs = list(amp_dir.glob('*'))
                if not len(turns_dirs):
                    raise OSError(f"No turns-dirs present in {str(amp_dir)}.")

                for turn_dir in turns_dirs:
                    if not turn_dir.is_dir():
                        continue

                    angle_dirs = list(turn_dir.glob('.*'))
                    if not len(angle_dirs):
                        raise OSError(f"No angle-dirs present in {str(turn_dir)}.")

                    for angle_dir in angle_dirs:
                        if not angle_dir.is_dir():
                            continue

                        htcondor_files = list(angle_dir.glob("htcondor.*"))
                        if len(htcondor_files) != 3:
                            raise OSError(f"Not all htcondor files present in {str(angle_dir)}.")

                        file_names = [f.name for f in angle_dir.glob("*")]
                        in_files_present = [f for f in SIXTRACK_INPUT_CHECK_FILES if f in file_names]
                        if len(in_files_present):
                            raise OSError(f"The files '{in_files_present}' are found in {str(angle_dir)},"
                                          "yet they should have been deleted after tracking.")

                        out_files_present = [f for f in SIXTRACK_OUTPUT_FILES if f in file_names]
                        if not len(out_files_present):
                            raise OSError(f"None of the expected output files '{SIXTRACK_OUTPUT_FILES}' "
                                          f"are present in {str(angle_dir)}")


# Helper -----------------------------------------------------------------------

def get_all_jobs_in_base(basedir):
    """ Returns all job-names in the sixdeskbase dir. """
    return [f.name.replace('workspace-', '') for f in basedir.glob("workspace-*")]


def get_first_dir(cwd: Path, glob: str = "*"):
    """ Return first directory of pattern `glob`. """
    for d in cwd.glob(glob):
        if d.is_dir():
            return d
    raise OSError(str(cwd))
