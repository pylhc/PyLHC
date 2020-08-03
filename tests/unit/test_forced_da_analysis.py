from pylhc.forced_da_analysis import main as fda_analysis

import inspect
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest


INPUT = Path(__file__).parent.parent / 'inputs'
DEBUG = False  # switch to local output instead of temp


class BasicTests:
    @staticmethod
    def test_md3312_data():
        data_dir = INPUT / 'kicks_vertical_md3312'
        with _output_dir() as output_dir:
            fda_analysis(
                beam=1,
                kick_directory=data_dir,
                energy=6500.,
                plane='Y',
                intensity_tfs=data_dir / 'intensity.tfs',
                emittance_tfs=data_dir / 'emittance_y.tfs',
                show_wirescan_emittance=data_dir / 'emittance_bws_y.tfs',
                output_directory=output_dir,
                # show=True,
            )
            check_output(output_dir, 'x')

    @staticmethod
    def test_md2162_timberdb():
        data_dir = INPUT / 'kicks_horizontal_md2162'
        with _output_dir() as output_dir:
            fda_analysis(
                fit='linear',
                beam=1,
                kick_directory=data_dir,
                energy=6500.,
                plane='X',
                output_directory=output_dir,
                pagestore_db=data_dir / 'MD2162_ACD_TimberDB_Fill6196.db',
                emittance_type='fit_sigma',
                show_wirescan_emittance=True,
                # show=True,
            )
            check_output(output_dir, 'x')


class ExtendedTests:
    @staticmethod
    def test_md3312_data_linear():
        data_dir = INPUT / 'kicks_vertical_md3312'
        with _output_dir() as output_dir:
            fda_analysis(
                fit='linear',
                beam=1,
                kick_directory=data_dir,
                energy=6500.,
                plane='Y',
                intensity_tfs=data_dir / 'intensity.tfs',
                emittance_tfs=data_dir / 'emittance_y.tfs',
                show_wirescan_emittance=data_dir / 'emittance_bws_y.tfs',
                output_directory=output_dir
            )
            check_output(output_dir, 'y')

    @staticmethod
    def test_md3312_no_data_given():
        with pytest.raises(OSError):
            with _output_dir() as output_dir:
                fda_analysis(
                    beam=1,
                    kick_directory=INPUT / 'kicks_vertical_md3312',
                    energy=6500.,
                    plane='Y',
                    output_directory=output_dir
                )


# Helper -----------------------------------------------------------------------


def _get_test_name():
    for s in inspect.stack():
        if s.function.startswith('test_'):
            return s.function
    raise AttributeError('Needs to be called downstream of a "test_" function')


@contextmanager
def _output_dir():
    if DEBUG:
        path = Path(f'temp_{_get_test_name()}')
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        yield path
    else:
        with tempfile.TemporaryDirectory() as dir_:
            yield Path(dir_)


def check_output(output_dir, plane):
    assert len(list(output_dir.glob('*.pdf'))) == 5
    assert len(list(output_dir.glob('*.tfs'))) == 4
    assert len(list(output_dir.glob('*.ini'))) == 1
    assert len(list(output_dir.glob(f'*_{plane}*'))) == 13
