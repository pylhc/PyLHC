"""
SixDesk Troubleshooting tools
-----------------------------

Some useful functions to troubleshoot the SixDesk output.
"""
from pathlib import Path

from omc3.utils import logging_tools

from pylhc.autosix import get_jobs_and_values
from pylhc.constants.autosix import (get_stagefile_path, Stage, get_track_path,
                                     get_workspace_path, SIXTRACK_INPUT_CHECK_FILES,
                                     SIXTRACK_OUTPUT_FILES, get_database_path)

LOG = logging_tools.get_logger(__name__)


# Stages -----------------------------------------------------------------------

def get_last_stage(jobname, basedir):
    """ Get the last run stage of job `jobname`. """
    stage_file = get_stagefile_path(jobname, basedir)
    last_stage = Stage[stage_file.read_text().strip('\n').split('\n')[-1]]
    return last_stage


# Set Stages ---

def set_stages_for_setup(basedir: Path, stage_name: str, jobid_mask: str, replace_dict: dict):
    """ Sets the last run stage for all jobs from given job-setups. """
    jobs, _ = get_jobs_and_values(jobid_mask, **replace_dict)
    for job in jobs:
        LOG.info(f"Setting stage to {stage_name} in {jobname}")
        set_stages(job, basedir, stage_name)


def set_stages(jobname: str, basedir: Path, stage_name: str):
    """ Sets the last run stage of all given jobs to `stage_name`. """
    if stage_name not in Stage.__members__:
        raise ValueError(f"Unknown stage '{stage_name}'")

    stage_file = get_stagefile_path(jobname, basedir)
    if stage_file.exists():
        stage_file.unlink()
    for stage in Stage:
        with open(stage_file, "a+") as f:
            f.write(f"{stage.name}\n")

        if stage.name == stage_name:
            break


def skip_stages(jobname: str, basedir: Path, stage_name: str):
    """ Skip stages until `stagename`, i.e. similar to `set_stages` but only if
    the stage hasn't been reached yet. Inverse to `reset_stages`"""
    if stage_name not in Stage.__members__:
        raise ValueError(f"Unknown stage '{stage_name}'")

    new_stage = Stage[stage_name]
    last_stage = get_last_stage(jobname, basedir)
    if last_stage < new_stage:
        LOG.info(f"Skipping stage form {last_stage.name} to {new_stage.name} in {jobname}")
        set_stages(jobname, basedir, stage_name)
    else:
        LOG.debug(f"Stage {last_stage.name} unchanged in {jobname}")


def reset_stages(jobname: str, basedir: Path, stagename: str):
    """ Reset stages until `stagename`, i.e. similar to `set_stages` but only if
    the stage has already been run. Inverse to `skip_stages`"""
    if stage_name not in Stage.__members__:
        raise ValueError(f"Unknown stage '{stage_name}'")

    new_stage = Stage[stage_name]
    last_stage = get_last_stage(jobname, basedir)
    if last_stage > new_stage:
        LOG.info(f"Resetting stage from {last_stage.name} to {new_stage.name} in {jobname}")
        set_stages(jobname, basedir, stage_name)
    else:
        LOG.debug(f"Stage {last_stage.name} unchanged in {jobname}")


# Check Stages ---

def check_stages_for_setup(basedir: Path, stage_name: str, jobid_mask: str, replace_dict: dict):
    """ Check the last run stage for all jobs from given job-setups. """
    jobs, _ = get_jobs_and_values(jobid_mask, **replace_dict)
    for job in jobs:
        check_last_stage(job, basedir)


def check_last_stage(jobname: str, basedir: Path):
    """ Logs names of all last run stages for given jobs. """
    last_stage = get_last_stage(jobname, basedir)
    LOG.info(f"'{jobname}' at stage '{last_stage.name}'")


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
            track = get_track_path(jobname=job, basedir=basedir)
            first_seed = get_first_dir(track) / 'simul'
            tunes = get_first_dir(first_seed)
            amp = get_first_dir(tunes, '*_*')
            turns = get_first_dir(amp, 'e*')
            angle = get_first_dir(turns, ".*")

            file_names = [f.name for f in angle.glob("*")]
            out_files_present = [f for f in SIXTRACK_OUTPUT_FILES if f in file_names]
            if not len(out_files_present):
                raise OSError(str(get_workspace_path(jobname=job, basedir=basedir)))
        except OSError as e:
            LOG.error(f"{e.args[0]} (stage: {get_last_stage(jobname=job, basedir=basedir).name})")
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

# Long Database Names Hack -----------------------------------------------------


def create_database_symlink(jobname: str, basedir: Path):
    db_path = get_database_path(jobname, basedir)
    if db_path.exists():
        LOG.debug(f"Database already exists in {jobname}.")
        return

    real_db_path = db_path.parent / "my.db"
    real_db_path.touch()

    db_path.symlink_to(real_db_path)
    LOG.info(f"Crated database link in {jobname}.")


def move_database_symlink(jobname: str, basedir: Path):
    db_path = get_database_path(jobname, basedir)
    real_db_path = db_path.parent / "my.db"
    if real_db_path.exists():
        real_db_path.rename(db_path)
        LOG.info(f"Renamed database to its proper name in {jobname}.")


# Helper -----------------------------------------------------------------------

def for_all_jobs(func: callable, basedir: Path, *args, **kwargs):
    """ Do function for all jobs in basedir. """
    for job in get_all_jobs_in_base(basedir):
        func(job, basedir, *args, **kwargs)


def get_all_jobs_in_base(basedir):
    """ Returns all job-names in the sixdeskbase dir. """
    return [f.name.replace('workspace-', '') for f in basedir.glob("workspace-*")]


def get_first_dir(cwd: Path, glob: str = "*"):
    """ Return first directory of pattern `glob`. """
    for d in cwd.glob(glob):
        if d.is_dir():
            return d
    raise OSError(str(cwd))
