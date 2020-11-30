from pathlib import Path
import numpy as np
import pandas as pd
import pathlib
import pytest
from pandas.testing import assert_series_equal

import tfs
from generic_parser.dict_parser import ArgumentError
from pylhc import bpm_calibration as calibration

INPUTS_DIR = Path(__file__).parent.parent / 'inputs' / 'calibration'
MEASUREMENTS_BETA = INPUTS_DIR / 'measurements' / 'for_beta'
MEASUREMENTS_DISPERSION = INPUTS_DIR / 'measurements' / 'for_dispersion'
MEASUREMENTS_SAME_BETA = INPUTS_DIR / 'measurements' / 'same_beta'
EXPECTED_OUTPUT = INPUTS_DIR / 'output'


def test_calibration_same_betabeat(tmp_path):
    factors = calibration.main(inputdir=MEASUREMENTS_BETA,
                               outputdir=tmp_path,
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

    # BetaBeat's tfs implementation is a bit different, we don't have the
    # same integer precision
    precision = 1e-14
    
    assert_series_equal(x_tfs['CALIBRATION'], expected_x_tfs['CALIBRATION'], atol=precision)
    assert_series_equal(y_tfs['CALIBRATION'], expected_y_tfs['CALIBRATION'], atol=precision)


def test_bad_args():
    with pytest.raises(ArgumentError) as e:
        calibration.main(inputdir='wat',
                         outputdir='',
                         ips=[1,5])

    assert "inputdir' is not of type Path" in str(e.value)


def test_no_beta_tfs(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        calibration.main(inputdir=pathlib.Path('wat'),
                         outputdir=tmp_path,
                         ips=[1,5])

    assert "No such file or directory:" in str(e.value)
    assert "beta_phase_x.tfs" in str(e.value)


def test_wrong_ip(tmp_path):
    with pytest.raises(ArgumentError) as e:
        calibration.main(inputdir=MEASUREMENTS_BETA,
                         outputdir=tmp_path,
                         ips=[15, 22])

    err = "All elements of 'ips' need to be one of '[1, 4, 5]', instead the list was [15, 22]."
    assert err in str(e.value)


def test_calibration_same_dispersion(tmp_path):
    factors = calibration.main(inputdir=MEASUREMENTS_DISPERSION,
                               outputdir=tmp_path,
                               method='dispersion',
                               ips=[1,5])

    # Let's open the tfs files we just created
    x_tfs = tfs.read(tmp_path / 'calibration_dispersion_x.tfs')
    
    # And the ones created by BetaBeat.src for the same measurements
    expected_x_tfs = tfs.read(EXPECTED_OUTPUT / 'calibration_dispersion_x.tfs')

    # Check all the BPMs are indeed the same 
    assert x_tfs['NAME'].equals(expected_x_tfs['NAME'])
    precision = 1e-4

    # BBsrc was wrong for the calibration error fit and the calibration fits
    # So we can only check the first column: CALIBRATION
    assert_series_equal(x_tfs['CALIBRATION'], expected_x_tfs['CALIBRATION'], atol=precision)


def test_beta_equal(tmp_path):
    factors = calibration.main(inputdir=MEASUREMENTS_SAME_BETA,
                               outputdir=tmp_path,
                               method='beta',
                               ips=[1,5])

    # beta from phase and beta amp are the same. Calibrations factors should
    # equal to 1
    expected = np.array([1.0] * len(factors['X']['CALIBRATION']))
    assert (factors['X']['CALIBRATION'].to_numpy() == expected).all()
    
    expected = np.array([1.0] * len(factors['Y']['CALIBRATION']))
    assert (factors['Y']['CALIBRATION'].to_numpy() == expected).all()

