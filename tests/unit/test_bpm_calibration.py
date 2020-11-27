from pathlib import Path
import numpy as np
import pandas as pd
import pathlib
import pytest

import tfs
from generic_parser.dict_parser import ArgumentError
from pylhc.bpm_calibration import calibration

INPUTS_DIR = Path(__file__).parent.parent / 'inputs' / 'calibration'
MEASUREMENTS_BETA = INPUTS_DIR / 'measurements' / 'for_beta'
MEASUREMENTS_DISPERSION = INPUTS_DIR / 'measurements' / 'for_dispersion'
EXPECTED_OUTPUT = INPUTS_DIR / 'output'


def test_calibration_same_betabeat(tmp_path):
    factors = calibration.main(input_path=MEASUREMENTS_BETA,
                               output_path=tmp_path,
                               ips=[1,4,5])

    # Let's open the tfs files we just created
    x_tfs = tfs.read(tmp_path / 'calibration_beta_x.tfs')
    y_tfs = tfs.read(tmp_path / 'calibration_beta_y.tfs')
    
    # And the ones created by BetaBeat.src for the same measurements
    expected_x_tfs = tfs.read(EXPECTED_OUTPUT / 'calibration_beta_x.tfs')
    expected_y_tfs = tfs.read(EXPECTED_OUTPUT / 'calibration_beta_y.tfs')

    # Check all the BPMs are indeed the same 
    assert x_tfs['NAME'].equals(expected_x_tfs['NAME'])
    assert y_tfs['NAME'].equals(expected_y_tfs['NAME'])

    # Drop the NAME column for float comparison
    x_tfs = x_tfs.drop(['NAME'], axis=1)
    y_tfs = y_tfs.drop(['NAME'], axis=1)
    expected_y_tfs = expected_y_tfs.drop(['NAME'], axis=1)
    expected_x_tfs = expected_x_tfs.drop(['NAME'], axis=1)

    # BetaBeat's tfs implementation is a bit different, we don't have the
    # same integer precision
    precision = 1e-14

    assert np.allclose(x_tfs, expected_x_tfs, atol=precision)
    assert np.allclose(y_tfs, expected_y_tfs, atol=precision)


def test_bad_args():
    with pytest.raises(ArgumentError) as e:
        calibration.main(input_path='wat',
                         output_path='',
                         ips=[1,5])

    assert "input_path' is not of type Path" in str(e.value)


def test_no_beta_tfs(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        calibration.main(input_path=pathlib.Path('wat'),
                         output_path=tmp_path,
                         ips=[1,5])

    assert "No such file or directory:" in str(e.value)
    assert "beta_phase_x.tfs" in str(e.value)


def test_wrong_ip(tmp_path):
    with pytest.raises(ArgumentError) as e:
        calibration.main(input_path=MEASUREMENTS_BETA,
                         output_path=tmp_path,
                         ips=[15, 22])

    err = "All elements of 'ips' need to be one of '[1, 4, 5]', instead the list was [15, 22]."
    assert err in str(e.value)


def test_calibration_same_dispersion(tmp_path):
    factors = calibration.main(input_path=MEASUREMENTS_DISPERSION,
                               output_path=tmp_path,
                               method='dispersion',
                               ips=[1,5])

    # Let's open the tfs files we just created
    x_tfs = tfs.read(tmp_path / 'calibration_dispersion_x.tfs')
    
    # And the ones created by BetaBeat.src for the same measurements
    expected_x_tfs = tfs.read(EXPECTED_OUTPUT / 'calibration_dispersion_x.tfs')

    # Check all the BPMs are indeed the same 
    assert x_tfs['NAME'].equals(expected_x_tfs['NAME'])

    # Drop the NAME column for float comparison
    x_tfs = x_tfs.drop(['NAME'], axis=1)
    expected_x_tfs = expected_x_tfs.drop(['NAME'], axis=1)

    precision = 1e-6

    # BBsrc was wrong for the calibration error fit and the calibration fits
    # So we can only check the calibration
    assert np.allclose(x_tfs['CALIBRATION'], expected_x_tfs['CALIBRATION'], atol=precision)

