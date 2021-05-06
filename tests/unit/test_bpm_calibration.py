from pathlib import Path
import numpy as np
import pandas as pd
import pathlib
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal

import tfs
from generic_parser.dict_parser import ArgumentError
from pylhc import bpm_calibration as calibration
from pylhc.constants.calibration import BPMS

INPUTS_DIR = Path(__file__).parent.parent / 'inputs' / 'calibration'
MEASUREMENTS = INPUTS_DIR / 'measurements'
EXPECTED_OUTPUT = INPUTS_DIR / 'output'


def test_calibration_same_betabeat(tmp_path):
    factors = calibration.main(inputdir=MEASUREMENTS / 'for_beta',
                               outputdir=tmp_path,
                               ips=[1, 4, 5])

    # Let's open the tfs files we just created
    x_tfs = tfs.read(tmp_path / 'calibration_beta_x.tfs', index='NAME')
    y_tfs = tfs.read(tmp_path / 'calibration_beta_y.tfs', index='NAME')

    # Those tfs need to be filtered because GetLLM only gives us the BPMs
    # used in ballistic optics
    x_tfs = x_tfs.reindex(BPMS[1][1] + BPMS[4][1] + BPMS[5][1])
    y_tfs = y_tfs.reindex(BPMS[1][1] + BPMS[4][1] + BPMS[5][1])
    
    # And the ones created by BetaBeat.src for the same measurements
    expected_x_tfs = tfs.read(EXPECTED_OUTPUT / 'calibration_beta_x.tfs', index='NAME')
    expected_y_tfs = tfs.read(EXPECTED_OUTPUT / 'calibration_beta_y.tfs', index='NAME')

    # BetaBeat's tfs implementation is a bit different, we don't have the
    # same integer precision
    precision = 1e-14

    # Drop the error calibration fit column because GetLLM fit was wrong
    tfs_ = [x_tfs, y_tfs, expected_x_tfs, expected_y_tfs]
    for i in range(len(tfs_)):
        tfs_[i] = tfs_[i].drop("ERROR_CALIBRATION_FIT", axis=1)
    
    # Compare the two dataframes
    assert_frame_equal(tfs_[0], tfs_[2], atol=precision)
    assert_frame_equal(tfs_[1], tfs_[3], atol=precision)


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
        calibration.main(inputdir=MEASUREMENTS / 'for_beta',
                         outputdir=tmp_path,
                         ips=[15, 22])

    err = "All elements of 'ips' need to be one of '[1, 4, 5]', instead the list was [15, 22]."
    assert err in str(e.value)


def test_calibration_same_dispersion(tmp_path):
    factors = calibration.main(inputdir=MEASUREMENTS / 'for_dispersion',
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
    factors = calibration.main(inputdir=MEASUREMENTS / 'same_beta',
                               outputdir=tmp_path,
                               method='beta')

    # beta from phase and beta amp are the same. Calibrations factors should
    # equal to 1
    expected = np.array([1.0] * len(factors['X']['CALIBRATION']))
    assert (factors['X']['CALIBRATION'].to_numpy() == expected).all()
    
    expected = np.array([1.0] * len(factors['Y']['CALIBRATION']))
    assert (factors['Y']['CALIBRATION'].to_numpy() == expected).all()


def test_missing_bpms(tmp_path):
    calibration.main(inputdir=MEASUREMENTS / 'missing_bpms',
                     outputdir=tmp_path,
                     method='beta',
                     ips=[1,5])

    factors = tfs.read(tmp_path / "calibration_beta_x.tfs", index="NAME")

    assert factors.loc["BPMWB.4R1.B1"]["CALIBRATION"] == 1
    assert factors.loc["BPMWB.4L1.B1"]["CALIBRATION"] == 1
    assert factors.loc["BPMS.2L1.B1"]["CALIBRATION"] != 1
    

def test_number_in_out(tmp_path):
    tfs_in = tfs.read(MEASUREMENTS / 'for_beta' / 'beta_phase_x.tfs')
    factors = calibration.main(inputdir=MEASUREMENTS / 'for_beta',
                               outputdir=tmp_path,
                               method='beta')

    assert len(factors["X"]) == len(tfs_in)


def test_no_error_tracking(tmp_path):
    # Test with tracking data on ballistic optics at IR4 without noise
    factors = calibration.main(inputdir=MEASUREMENTS / 'tracking',
                               outputdir=tmp_path,
                               ips=[4])

    x_df = factors['X'].reset_index(drop=True)
    y_df = factors['Y'].reset_index(drop=True)
    ir4_x_df = factors['X'].reindex(BPMS[4][1]).reset_index(drop=True)
    ir4_y_df = factors['X'].reindex(BPMS[4][1]).reset_index(drop=True)
    precision = 1e-3
    
    # All factors ≃ 1
    expected = pd.Series([1.0] * len(factors['X']['CALIBRATION']))
    assert_series_equal(x_df['CALIBRATION'], expected, atol=precision, check_names=False)
    assert_series_equal(y_df['CALIBRATION'], expected, atol=precision, check_names=False)
    
    # And their error ≃
    expected = pd.Series([0.0] * len(factors['X']['CALIBRATION']))
    assert_series_equal(x_df['ERROR_CALIBRATION'], expected, atol=precision, check_names=False)
    assert_series_equal(y_df['ERROR_CALIBRATION'], expected, atol=precision, check_names=False)

    # Same with fit
    expected = pd.Series([1.0] * len(ir4_x_df['CALIBRATION_FIT']))
    assert_series_equal(ir4_x_df['CALIBRATION_FIT'], expected, atol=precision, check_names=False)
    assert_series_equal(ir4_y_df['CALIBRATION_FIT'], expected, atol=precision, check_names=False)
    
    # and its errors
    expected = pd.Series([0.0] * len(ir4_x_df['ERROR_CALIBRATION_FIT']))
    assert_series_equal(ir4_x_df['ERROR_CALIBRATION_FIT'], expected, atol=precision, check_names=False)
    assert_series_equal(ir4_y_df['ERROR_CALIBRATION_FIT'], expected, atol=precision, check_names=False)


