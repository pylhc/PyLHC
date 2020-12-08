import logging
from pathlib import Path
from unittest.mock import patch

import tfs

from pylhc.sixdesk_tools.post_process_da import plot_polar, plt
from pylhc.autosix import _generate_jobs, setup_and_run
from pylhc.constants.autosix import (
    get_masks_path, get_autosix_results_path, get_sixdeskenv_path,
    get_sysenv_path, get_stagefile_path, STAGE_ORDER, ANGLE, STAGES
)


INPUTS = Path(__file__).parent.parent / 'inputs'
DA_RESULTS_DIR = INPUTS / 'sixdesk_da_results'


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

    def subprocess_mock(*args, **kwargs):
        tmp_path.mkdir(exist_ok=True, parents=True)
        get_masks_path(jobname, tmp_path).mkdir(exist_ok=True, parents=True)

    with patch('pylhc.sixdesk_tools.create_workspace.start_subprocess', new=subprocess_mock),\
         patch('pylhc.sixdesk_tools.submit.start_subprocess', new=subprocess_mock):
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
        assert all(s in stagefile_text for s in STAGE_ORDER[:3])


def test_create_workspace_stop_init(tmp_path):
    jobname="test_job"

    def subprocess_mock(*args, **kwargs):
        tmp_path.mkdir(exist_ok=True, parents=True)
        get_masks_path(jobname, tmp_path).mkdir(exist_ok=True, parents=True)

    with patch('pylhc.sixdesk_tools.create_workspace.start_subprocess', new=subprocess_mock), \
         patch('pylhc.sixdesk_tools.submit.start_subprocess', new=subprocess_mock):
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
        assert STAGES.create_job in stagefile_text
        assert STAGES.initialize_workspace not in stagefile_text


def test_skip_all_stages(tmp_path, caplog):
    jobname = "test_job"
    stagefile = get_stagefile_path(jobname, tmp_path)
    stagefile.parent.mkdir(parents=True)
    stagefile.write_text("\n".join(STAGE_ORDER[:-1]))
    with caplog.at_level(logging.INFO):
        setup_and_run(jobname, tmp_path)

    assert all(stage in caplog.text for stage in STAGE_ORDER[:-1])
    assert "All stages run." in caplog.text


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

