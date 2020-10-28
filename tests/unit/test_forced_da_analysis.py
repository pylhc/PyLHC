from pathlib import Path

import matplotlib
import pytest

from pylhc.forced_da_analysis import main as fda_analysis

# Forcing non-interactive Agg backend so rendering is done similarly across platforms during tests
matplotlib.use("Agg")

INPUT = Path(__file__).parent.parent / 'inputs'


@pytest.mark.cern_network
class TestOnCernNetwork:
    def test_md3312_data(self, tmp_path):
        data_dir = INPUT / 'kicks_vertical_md3312'
        fda_analysis(
            beam=1,
            kick_directory=data_dir,
            energy=6500.,
            plane='Y',
            intensity_tfs=data_dir / 'intensity.tfs',
            emittance_tfs=data_dir / 'emittance_y.tfs',
            show_wirescan_emittance=data_dir / 'emittance_bws_y.tfs',
            output_directory=tmp_path,
            # show=True,
        )
        check_output(tmp_path)

    def test_md3312_data_linear(self, tmp_path):
        data_dir = INPUT / 'kicks_vertical_md3312'
        fda_analysis(
            fit='linear',
            beam=1,
            kick_directory=data_dir,
            energy=6500.,
            plane='Y',
            intensity_tfs=data_dir / 'intensity.tfs',
            emittance_tfs=data_dir / 'emittance_y.tfs',
            show_wirescan_emittance=data_dir / 'emittance_bws_y.tfs',
            output_directory=tmp_path
        )
        check_output(tmp_path)

    def test_md3312_no_data_given(self, tmp_path):
        with pytest.raises(OSError):
            fda_analysis(
                beam=1,
                kick_directory=INPUT / 'kicks_vertical_md3312',
                energy=6500.,
                plane='Y',
                output_directory=tmp_path
            )


def test_md2162_timberdb(tmp_path):
    data_dir = INPUT / 'kicks_horizontal_md2162'
    fda_analysis(
        fit='linear',
        beam=1,
        kick_directory=data_dir,
        energy=6500.,
        plane='X',
        output_directory=tmp_path,
        pagestore_db=data_dir / 'MD2162_ACD_TimberDB_Fill6196.db',
        emittance_type='fit_sigma',
        show_wirescan_emittance=True,
        # show=True,
    )
    check_output(tmp_path)


# Helper -----------------------------------------------------------------------


def check_output(output_dir: Path) -> None:
    assert len(list(output_dir.glob('*.pdf'))) == 5
    assert len(list(output_dir.glob('*.tfs'))) == 4
    assert len(list(output_dir.glob('*.ini'))) == 1
    assert len(list(output_dir.glob('*_[xy]*'))) == 13
