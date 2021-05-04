import sys
from pathlib import Path

import pytest
from generic_parser import DotDict

from pylhc.job_submitter import main as job_submit

skip_not_linux = pytest.mark.skipif(
    sys.platform != "linux", reason="htcondor python bindings from PyPI are only on linux"
)

skip_on_linux = pytest.mark.skipif(
    sys.platform == "linux", reason="htcondor python bindings are present"
)


@skip_not_linux
def test_job_creation_and_localrun(tmp_path):
    args, setup = _create_setup(tmp_path)
    setup.update(run_local=True)
    job_submit(**setup)
    _test_output(args)


@skip_not_linux
def test_find_errornous_percentage_signs(tmp_path):
    mask = "%(PARAM1)s.%(PARAM2)d\nsome stuff # should be 5%\nsome % more % stuff."
    args, setup = _create_setup(tmp_path, mask_content=mask)
    with pytest.raises(KeyError) as e:
        job_submit(**setup)
    assert "problematic '%'" in e.value.args[0]


@skip_not_linux
def test_missing_keys(tmp_path):
    mask = "%(PARAM1)s.%(PARAM2)s.%(PARAM3)s"
    args, setup = _create_setup(tmp_path, mask_content=mask)
    with pytest.raises(KeyError) as e:
        job_submit(**setup)
    assert "PARAM3" in e.value.args[0]


@skip_on_linux
def test_not_on_linux(tmp_path):
    args, setup = _create_setup(tmp_path)
    with pytest.raises(EnvironmentError) as e:
        job_submit(**setup)
    assert "htcondor bindings" in e.value.args[0]


@skip_not_linux
@pytest.mark.cern_network
def test_htc_submit():
    """ This test is here for local testing only. You need to adapt the path
    and delete the results afterwards manually (so you can check them before."""
    user = "jdilly"
    path = Path("/", "afs", "cern.ch", "user", user[0], user, "htc_temp")
    path.mkdir(exist_ok=True)
    args, setup = _create_setup(path)

    job_submit(**setup)
    _test_output(args, post_run=False)
    # _test_output(args, post_run=True)  # you can use this if you like after htcondor is done


# Helper -----------------------------------------------------------------------


def _create_setup(cwd_path: Path, mask_content: str = None):
    """ Create a quick setup for Parameters PARAM1 and PARAM2. """
    out_name = "out.txt"
    out_dir = "Outputdir"

    args = DotDict(
        cwd=cwd_path,
        out_name=out_name,
        out_dir=out_dir,
        id="%(PARAM1)s.%(PARAM2)d",
        mask_name="test_script.mask",
        ext=".sh",
        out_file=Path(out_dir, out_name),
        p1_list=["a", "b"],
        p2_list=[1, 2, 3],
    )

    mask_path = cwd_path / args.mask_name
    if mask_content is None:
        mask_content = args.id

    with mask_path.open("w") as f:
        f.write(f'echo "{mask_content}" > "{args.out_file}"\n')

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
    return args, setup


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
            # assert out_dir_path.exists()  # does not seem to be pre-created anymore (jdilly 2021-05-04)
            if post_run:
                assert out_dir_path.is_dir()
                assert out_file_path.exists()
                assert out_file_path.is_file()

                with out_file_path.open("r") as f:
                    assert f.read().strip("\n") == current_id
