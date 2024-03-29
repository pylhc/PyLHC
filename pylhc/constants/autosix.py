"""
Constants: Autosix
----------------------------------

Collections of constants and paths used in autosix.

:module: constants.autosix
:author: jdilly

"""
from pathlib import Path

from generic_parser import DotDict

from pylhc.constants.external_paths import SIXDESK_UTILS, MADX_BIN

# Program Paths ----------------------------------------------------------------
SETENV_SH = SIXDESK_UTILS / "set_env.sh"
MAD_TO_SIXTRACK_SH = SIXDESK_UTILS / "mad6t.sh"
RUNSIX_SH = SIXDESK_UTILS / "run_six.sh"
RUNSTATUS_SH = SIXDESK_UTILS / "run_status"
DOT_PROFILE = SIXDESK_UTILS / "dot_profile"
SIXDB = SIXDESK_UTILS.parent / "externals" / "SixDeskDB" / "sixdb"
SIXDESKLOCKFILE = "sixdesklock"

# Constants and Requirements ---------------------------------------------------

HEADER_BASEDIR = "BASEDIR"

DEFAULTS = dict(
    python="python3",
    da_turnstep=100,
    executable=MADX_BIN,
)

# Sixenv ---
SIXENV_REQUIRED = ["TURNS", "AMPMIN", "AMPMAX", "AMPSTEP", "ANGLES"]
SIXENV_DEFAULT = dict(
    RESUBMISSION=0,  # 0: never, 1: if fort.10, 2: always
    PLATFORM="HTCondor",
    LOGLEVEL=0,  # 0: basic + errors , 1: + info, >=2: + debug
    FIRSTSEED=1,
    LASTSEED=60,
    ENERGY="col",  # 'col' or 'inj'
    NPAIRS=30,  # 1-32 particle pairs
    EMITTANCE=3.75,  # normalized emittance
    DIMENSIONS=6,  # Phase-Space dimensions
    WRITEBINS=500,
)
SEED_KEYS = ["FIRSTSEED", "LASTSEED"]


# Stages ---
STAGE_ORDER = [
    "create_job",
    "initialize_workspace",
    "submit_mask",
    "check_input",
    "submit_sixtrack",
    "check_sixtrack_output",
    "sixdb_load",
    "sixdb_cmd",
    "post_process",
    "final",
]
STAGES = DotDict({key: key for key in STAGE_ORDER})


# SixDB and Postprocess ---

HEADER_NTOTAL, HEADER_INFO, HEADER_HINT = "NTOTAL", "INFO", "HINT"
MEAN, STD, MIN, MAX, N = "MEAN", "STD", "MIN", "MAX", "N"
SEED, ANGLE, ALOST1, ALOST2, AMP = "SEED", "ANGLE", "ALOST1", "ALOST2", "A"


# Workspace Paths --------------------------------------------------------------

# Input ---
def get_workspace_path(jobname: str, basedir: Path) -> Path:
    return basedir / f"workspace-{jobname}"


def get_scratch_path(basedir: Path) -> Path:
    return basedir / f"scratch-0"


def get_sixjobs_path(jobname: str, basedir: Path) -> Path:
    return get_workspace_path(jobname, basedir) / "sixjobs"


def get_sixdeskenv_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / "sixdeskenv"


def get_sysenv_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / "sysenv"


def get_masks_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / "mask"


def get_database_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / f"{jobname}.db"


def get_sixtrack_input_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / "sixtrack_input"


def get_mad6t_mask_path(jobname: str, basedir: Path) -> Path:
    return get_sixtrack_input_path(jobname, basedir) / "mad6t.sh"


def get_mad6t1_mask_path(jobname: str, basedir: Path) -> Path:
    return get_sixtrack_input_path(jobname, basedir) / "mad6t1.sh"


# Output ---


def get_autosix_results_path(jobname: str, basedir: Path) -> Path:
    return get_sixjobs_path(jobname, basedir) / "autosix_output"


def get_stagefile_path(jobname: str, basedir: Path) -> Path:
    return get_autosix_results_path(jobname, basedir) / "stages_completed.txt"


def get_tfs_da_path(jobname: str, basedir: Path) -> Path:
    return get_autosix_results_path(jobname, basedir) / f"{jobname}_da.tfs"


def get_tfs_da_seed_stats_path(jobname: str, basedir: Path) -> Path:
    return get_autosix_results_path(jobname, basedir) / f"{jobname}_da_per_seed.tfs"


def get_tfs_da_angle_stats_path(jobname: str, basedir: Path) -> Path:
    return get_autosix_results_path(jobname, basedir) / f"{jobname}_da_per_angle.tfs"
