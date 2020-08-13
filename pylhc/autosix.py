from pathlib import Path
from omc3.utils import logging_tools
from pylhc.sixdesk_tools.create_workspace import create_jobs
from pylhc.sixdesk_tools.submit import submit_mask, submit_sixtrack
from pylhc.sixdesk_tools.utils import is_locked

LOG = logging_tools.get_logger(__name__, level_console=logging_tools.DEBUG)


def main():
    jobname = "testjob"
    basedir = Path('/afs/cern.ch/work/j/jdilly/sixdeskbase')
    ssh = 'lxplus'
    unlock = True
    with open(Path('/home/jdilly/Work/study.20.irnl_correction_with_feeddown/code.full_study/python_mask.py'), 'r') as mask_f:
        mask = mask_f.read()

    if is_locked(jobname, basedir, unlock=unlock):
        LOG.info(f"{jobname} is locked. Aborting.")
        return

    create_jobs(jobname, basedir, mask,
                ssh=ssh,
                binary_path=Path('/afs/cern.ch/work/j/jdilly/public/venvs/for_htc/bin/python'),
                BEAM=1,
                TURNS=100000,
                AMPMIN=2, AMPMAX=20, AMPSTEP=5,
                ANGLES=50,
                B6viaB4=False,
                )

    submit_mask(jobname, basedir, ssh=ssh)
    submit_sixtrack(jobname, basedir, ssh=ssh)


if __name__ == '__main__':
    main()

