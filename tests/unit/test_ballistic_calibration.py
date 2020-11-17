from pathlib import Path
import numpy as np
import pandas as pd
import pathlib
import pytest

import tfs
from generic_parser.dict_parser import ArgumentError
from pylhc import BPM_calibration as calibration

INPUTS_DIR = Path(__file__).parent.parent / 'inputs' / 'calibration'
MEASUREMENTS_BETA = INPUTS_DIR / 'measurements' / 'for_beta'
MEASUREMENTS_DISPERSION = INPUTS_DIR / 'measurements' / 'for_dispersion'
MODEL = INPUTS_DIR / 'model'
EXPECTED_OUTPUT = INPUTS_DIR / 'output'


def test_calibration_same_betabeat(tmp_path):
    factors = calibration.main(input_path=MEASUREMENTS_BETA,
                               model_path=MODEL,
                               output_path=tmp_path)

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
                         model_path='Cake',
                         output_path='')

    assert "input_path' is not of type Path" in str(e.value)


def test_no_beta_tfs(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        calibration.main(input_path=pathlib.Path('wat'),
                         model_path=MODEL,
                         output_path=tmp_path)

    assert "File beta_phase_x.tfs couldn't be found in directory" in str(e.value)


def test_no_model_tfs(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        calibration.main(input_path=MEASUREMENTS_BETA,
                         model_path=pathlib.Path('oopsie'),
                         output_path=tmp_path)
    
    assert "No such file or directory" in str(e.value)


def test_bad_beam():
    model = tfs.TfsDataFrame()
    model.SEQUENCE = 'LHCB4'

    with pytest.raises(ValueError) as e:
        calibration._get_beam_from_model(model)

    assert "Could not find a correct value for beam in model: 4" in str(e.value)



