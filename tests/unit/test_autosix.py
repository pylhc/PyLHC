import logging
from pathlib import Path
from unittest.mock import patch

import tfs

from pylhc.sixdesk_tools.create_workspace import remove_twiss_fail_check
from pylhc.sixdesk_tools.post_process_da import plot_polar, plt
from pylhc.autosix import _generate_jobs, setup_and_run
from pylhc.constants.autosix import (
    get_masks_path, get_autosix_results_path, get_sixdeskenv_path,
    get_sysenv_path, get_stagefile_path, ANGLE, Stage, get_mad6t_mask_path, get_mad6t1_mask_path
)

STAGE_NAMES = [s.name for s in Stage]

INPUTS = Path(__file__).parent.parent / 'inputs'
DA_RESULTS_DIR = INPUTS / 'sixdesk_da_results'

# log-texts to be changed if they change in `should_run_stage`
ALREADY_RUN_LOG = "Stage {} has already been run."
PREVIOUS_MISSING_LOG = "Stage {} not run because previous stage(s) missing."
AFTER_MAXIMUM_LOG = "Stage {} would run after requested maximum stage {}."


def test_create_job_matrix(tmp_path):
    jobs_df = _generate_jobs(tmp_path, jobid_mask=None, 
                             param0=[1, 2., 3],
                             param1=[4],
                             param2=["test", "some", "more"]
                             )
    
    assert tmp_path in jobs_df.headers.values()
    assert len(jobs_df.index) == 9
    assert all(f"param{i}" in jobs_df.columns for i in range(3))
    assert len(list(tmp_path.glob("*.tfs"))) == 1


def test_create_workspace(tmp_path):
    jobname="test_job"

    mock_create, mock_submit = _create_subprocess_mocks(jobname, tmp_path)
    with mock_create, mock_submit:
        setup_and_run(jobname=jobname,
                      basedir=tmp_path,
                      mask_text="Just a mask %(PARAM1)s %(PARAM2)s %(BEAM)s",
                      binary_path=Path('somethingcomplicated/pathomatic'),
                      PARAM1=4,
                      PARAM2="%SEEDRAN",
                      BEAM=1,
                      TURNS=10101,
                      AMPMIN=2, AMPMAX=20, AMPSTEP=2,
                      ANGLES=5,
                      )

        mask = next(get_masks_path(jobname, tmp_path).glob("*"))
        assert mask.exists()

        mask_text = mask.read_text()
        assert "%SEEDRAN" in mask_text
        assert "4" in mask_text
        assert "1" in mask_text
        assert "mask" in mask_text

        sixdeskenv = get_sixdeskenv_path(jobname, tmp_path)
        assert sixdeskenv.exists()

        sixdeskenv_text = sixdeskenv.read_text()
        assert "10101" in sixdeskenv_text

        sysenv = get_sysenv_path(jobname, tmp_path)
        assert sysenv.exists()

        sysenv_text = sysenv.read_text()
        assert "somethingcomplicated" in sysenv_text
        assert "pathomatic" in sysenv_text

        autosix_result = get_autosix_results_path(jobname, tmp_path)
        assert autosix_result.exists()

        stagefile = get_stagefile_path(jobname, tmp_path)
        assert stagefile.exists()

        stagefile_text = stagefile.read_text()
        assert all(s in stagefile_text for s in STAGE_NAMES[:3])


def test_create_workspace_stop_init(tmp_path):
    jobname="test_job"

    mock_create, mock_submit = _create_subprocess_mocks(jobname, tmp_path)
    with mock_create, mock_submit:
        setup_and_run(jobname=jobname,
                      basedir=tmp_path,
                      binary_path=Path('somethingcomplicated/pathomatic'),
                      stop_workspace_init=True,
                      mask_text="",
                      TURNS=10101,
                      AMPMIN=2, AMPMAX=20, AMPSTEP=2,
                      ANGLES=5,
                      FIRSTSEED=None, LASTSEED=None,
                      )

        stagefile = get_stagefile_path(jobname, tmp_path)
        assert stagefile.exists()

        stagefile_text = stagefile.read_text()
        assert Stage.create_job.name in stagefile_text
        assert Stage.initialize_workspace.name not in stagefile_text


def test_skip_all_stages(tmp_path, caplog):
    """ Skips all stages but the last one, which prints "All stages run". """
    jobname = "test_job"

    stagefile = get_stagefile_path(jobname, tmp_path)
    stagefile.parent.mkdir(parents=True)
    stagefile.write_text("\n".join(STAGE_NAMES[:-1]))
    with caplog.at_level(logging.INFO):
        setup_and_run(jobname, tmp_path)

    assert all(ALREADY_RUN_LOG.format(s) in caplog.text for s in STAGE_NAMES[:-1])
    assert "All stages run." in caplog.text


def test_max_stage(tmp_path, caplog):
    """ Skips all stages, first because they had already 'run',
    the others because they come after `max_stage`. """
    jobname = "test_job"
    run_stages = STAGE_NAMES[:-2]  # ends at `-3`
    max_stage = STAGE_NAMES[-3]
    after_max_stages = STAGE_NAMES[-2:]

    stagefile = get_stagefile_path(jobname, tmp_path)
    stagefile.parent.mkdir(parents=True)
    stagefile.write_text("\n".join(run_stages))
    with caplog.at_level(logging.INFO):
        setup_and_run(jobname, tmp_path, max_stage=max_stage)

    assert all(ALREADY_RUN_LOG.format(s) in caplog.text for s in run_stages)
    assert all(AFTER_MAXIMUM_LOG.format(s, max_stage) in caplog.text for s in after_max_stages)
    assert not any(PREVIOUS_MISSING_LOG.format(s) in caplog.text for s in STAGE_NAMES)


def test_polar_plot(tmp_path):
    df_angles = tfs.read(DA_RESULTS_DIR / 'da_per_angle.tfs', index=ANGLE)
    df_da = tfs.read(DA_RESULTS_DIR / 'da.tfs')
    fig = plot_polar(df_angles=df_angles, df_da=df_da, interpolated=False, fill=True)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 63  # 60 Seeds, MEAN, MIN, MAX
    # plt.show()


def test_polar_plot_interpolated(tmp_path):
    df_angles = tfs.read(DA_RESULTS_DIR / 'da_per_angle.tfs', index=ANGLE)
    df_da = tfs.read(DA_RESULTS_DIR / 'da.tfs')
    fig = plot_polar(df_angles=df_angles, df_da=df_da, interpolated=True, fill=False)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 63  # 60 Seeds, MEAN, MIN, MAX
    # plt.show()


def test_twissfail_removal(tmp_path):
    jobname = "test_job"
    mad6t, mad6t1 = _create_mad6t_files(jobname, tmp_path)
    remove_twiss_fail_check(jobname, tmp_path)

    for f in (mad6t, mad6t1):
        mad6t_lines = f.read_text().split("\n")

        assert all(l.startswith("#") for l in mad6t_lines[:-1])
        assert mad6t_lines[-1].startswith("if")


# Helper -----------------------------------------------------------------------


def _create_subprocess_mocks(jobname, dirpath):
    def subprocess_mock(*args, **kwargs):
        dirpath.mkdir(exist_ok=True, parents=True)
        get_masks_path(jobname, dirpath).mkdir(exist_ok=True, parents=True)

    mock_crate = patch('pylhc.sixdesk_tools.create_workspace.start_subprocess', new=subprocess_mock)
    mock_submit = patch('pylhc.sixdesk_tools.submit.start_subprocess', new=subprocess_mock)
    return mock_crate, mock_submit


def _create_mad6t_files(jobname, tmppath):
    mad6t = get_mad6t_mask_path(jobname, tmppath)
    mad6t1 = get_mad6t1_mask_path(jobname, tmppath)
    mad6t.parent.mkdir(parents=True)
    mad6t.write_text(_mad6t_text())
    mad6t1.write_text(_mad6t_text())
    return mad6t, mad6t1


def _mad6t_text():
    return """grep -i "TWISS fail" $filejob.out."$i" > /dev/null
if test $? -eq 0
then
  touch $sixtrack_input/ERRORS
  echo "MADX TWISS appears to have failed!"
  echo "$filejob.out.${i} MADX TWISS appears to have failed!" >> $sixtrack_input/ERRORS
  exit 2
fi
if test ! -s fc.2"""
