import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from generic_parser import DotDict

try:
    from pylhc.job_submitter import main as job_submit
except SystemExit:  # might have triggered exit() because htcondor not found if not on linux
    pass  # let the skipif marker take care if the rest


@pytest.mark.skipif(
    sys.platform != "linux", reason="htcondor python bindings from PyPI are only on linux"
)
class TestHTCondorSubmitter:
    def test_job_creation_and_localrun(self):
        with _create_setup() as (args, setup):
            setup.update(run_local=True)
            job_submit(**setup)
            _test_output(args)

    @pytest.mark.cern_network
    def test_htc_submit(self):
        """ This test is here for local testing only. You need to adapt the path
        and delete the results afterwards manually (so you can check them before."""
        user = "jdilly"
        path = Path("/", "afs", "cern.ch", "user", user[0], user, "htc_temp")

        try:
            path.mkdir(exist_ok=True)
        except IOError:
            return
        else:
            with _create_setup(path) as (args, setup):
                job_submit(**setup)
                _test_output(args, post_run=False)
        # _test_output(args, post_run=True)  # you can use this if you like after htcondor is done


# Helper -----------------------------------------------------------------------


@contextmanager
def _create_setup(afs_path: Path = None):
    out_name = "out.txt"
    out_dir = "Outputdir"

    with tempfile.TemporaryDirectory() as cwd:
        cwd = Path(cwd)
        if afs_path is not None:
            cwd = afs_path

        args = DotDict(
            cwd=cwd,
            out_name=out_name,
            out_dir=out_dir,
            id="%(PARAM1)s.%(PARAM2)d",
            mask_name="test_script.mask",
            ext=".sh",
            out_file=Path(out_dir, out_name),
            p1_list=["a", "b"],
            p2_list=[1, 2, 3],
        )

        mask_path = cwd / args.mask_name
        with mask_path.open("w") as f:
            f.write(f'echo "{args.id}" > "{args.out_file}"\n')

        setup = dict(
            executable="/bin/bash",
            script_extension=args.ext,
            job_output_dir=out_dir,
            mask=str(mask_path),
            replace_dict=dict(PARAM1=args.p1_list, PARAM2=args.p2_list,),
            jobid_mask=args.id,
            jobflavour="workday",
            resume_jobs=True,
            check_files=[args.out_name],
            working_directory=str(args.cwd),
            dryrun=False,
        )
        yield args, setup


def _test_output(args, post_run=True):
    for p1 in args.p1_list:
        for p2 in args.p2_list:
            current_id = args.id % dict(PARAM1=p1, PARAM2=p2)
            job_name = f"Job.{current_id}"
            job_dir_path = args.cwd / job_name
            out_dir_path = job_dir_path / args.out_dir
            out_file_path = out_dir_path / args.out_name

            assert job_dir_path.exists()
            assert job_dir_path.is_dir()
            assert (job_dir_path / args.mask_name).with_suffix(args.ext).exists()
            assert out_dir_path.exists()
            if post_run:
                assert out_dir_path.is_dir()
                assert out_file_path.exists()
                assert out_file_path.is_file()

                with out_file_path.open("r") as f:
                    assert f.read().strip("\n") == current_id
