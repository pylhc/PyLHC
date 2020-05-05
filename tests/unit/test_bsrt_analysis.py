import os
from ast import literal_eval
import pandas as pd
import pytest
from pylhc import BSRT_analysis

CURRENT_DIR = os.path.dirname(__file__)

INPUTS = os.path.join(CURRENT_DIR, os.pardir, 'inputs', 'bsrt_analysis')


def test_bsrt_df(_bsrt_df):
    results = BSRT_analysis.main(directory=INPUTS, beam='B1')
    pd.testing.assert_frame_equal(results['bsrt_df'], _bsrt_df)


@pytest.mark.mpl_image_compare
def test_fitvarplot(_bsrt_df):
    return BSRT_analysis.plot_fit_variables({'show_plots': False, 'outputdir': None}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_fullcrossection(_bsrt_df):
    return BSRT_analysis.plot_full_crosssection({'show_plots': False, 'outputdir': None}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_auxiliary_variables(_bsrt_df):
    return BSRT_analysis.plot_auxiliary_variables({'show_plots': False, 'outputdir': None}, _bsrt_df)


@pytest.fixture()
def _bsrt_df():
    return pd.read_csv(os.path.join(INPUTS, BSRT_analysis._get_bsrt_tfs_fname('B1')),
                       parse_dates=True,
                       index_col='TimeIndex',
                       quotechar='"',
                       converters={"acquiredImageRectangle": literal_eval,
                                   "beam": literal_eval,
                                   "gateMode": literal_eval,
                                   "imageSet": literal_eval,
                                   "lastFitResults": literal_eval,
                                   "projDataSet1": literal_eval,
                                   "projDataSet2": literal_eval,
                                   "projPositionSet1": literal_eval,
                                   "projPositionSet2": literal_eval
                                   }
                                   )
