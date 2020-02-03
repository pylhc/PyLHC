import tempfile
import pytest
from pathlib import Path
import os

from pylhc.job_submitter import main as job_submit


def test_job_creation_and_run():
    out_dir = "Outputdir"
    id_ = '%(PARAM1)s.%(PARAM2)d'
    out_name = 'out.txt'
    mask_name = 'test_script.mask'
    ext = 'sh'
    out_file = Path(out_dir) / out_name

    p1_list = ['a', 'b']
    p2_list = [1, 2, 3]

    with tempfile.TemporaryDirectory() as cwd:
        cwd = Path(cwd)
        mask_path = cwd / mask_name
        with mask_path.open("w") as f:
            f.write(f'echo "{id_}" > "{out_file}"\n')

        job_submit(
            executable='/bin/bash',
            script_extension=ext,
            job_output_dir=out_dir,
            mask=str(mask_path),
            replace_dict=dict(
                PARAM1=p1_list,
                PARAM2=p2_list,
            ),
            jobid_mask=id_,
            jobflavour="workday",
            resume_jobs=True,
            check_files=[out_name],
            run_local=True,
            working_directory=str(cwd),
            dryrun=False,
        )

        for p1 in p1_list:
            for p2 in p2_list:
                current_id = id_ % dict(PARAM1=p1, PARAM2=p2)
                job_name = f"Job.{current_id}"
                job_dir_path = cwd / job_name
                out_dir_path = job_dir_path / out_dir
                out_file_path = out_dir_path / out_name

                assert job_dir_path.exists()
                assert job_dir_path.is_dir()
                assert (job_dir_path / mask_name).with_suffix("." + ext).exists()
                assert out_dir_path.exists()
                assert out_dir_path.is_dir()
                assert out_file_path.exists()
                assert out_file_path.is_file()

                with out_file_path.open("r") as f:
                    assert f.read().strip('\n') == current_id
