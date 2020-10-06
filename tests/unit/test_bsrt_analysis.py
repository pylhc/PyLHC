import os
from ast import literal_eval

import numpy as np
import pandas as pd
import pytest

from pylhc import BSRT_analysis

CURRENT_DIR = os.path.dirname(__file__)
INPUTS = os.path.join(CURRENT_DIR, os.pardir, 'inputs', 'bsrt_analysis')


def test_bsrt_df(_bsrt_df):
    results = BSRT_analysis.main(directory=INPUTS, beam='B1')
    pd.testing.assert_frame_equal(results['bsrt_df'], _bsrt_df)


def test_select_by_time():
    time_df = pd.DataFrame(index=np.linspace(0, 10, 11),
                           data={'A': np.linspace(0, 10, 11)})
    with pytest.raises(AssertionError):
        BSRT_analysis._select_files({'starttime':3, 'endtime':1}, time_df)
    pd.testing.assert_frame_equal(BSRT_analysis._select_files({'starttime':1, 'endtime':3}, time_df),
                                  time_df.loc[1:3])
    pd.testing.assert_frame_equal(BSRT_analysis._select_files({'starttime':1, 'endtime':None}, time_df),
                                  time_df.loc[1:])
    pd.testing.assert_frame_equal(BSRT_analysis._select_files({'starttime':None, 'endtime':3}, time_df),
                                  time_df.loc[:3])

@pytest.mark.mpl_image_compare
def test_fitvarplot(_bsrt_df):
    return BSRT_analysis.plot_fit_variables({'show_plots': False,
                                             'outputdir': None,
                                             'kick_df': None}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_fitvarplot_with_kick_df(_bsrt_df, _kick_df):
    return BSRT_analysis.plot_fit_variables({'show_plots': False,
                                             'outputdir': None,
                                             'kick_df': _kick_df}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_fullcrossection(_bsrt_df):
    return BSRT_analysis.plot_full_crosssection({'show_plots': False,
                                                 'outputdir': None,
                                                 'kick_df': None}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_fullcrossection_with_kick_df(_bsrt_df, _kick_df):
    return BSRT_analysis.plot_full_crosssection({'show_plots': False,
                                                 'outputdir': None,
                                                 'kick_df': _kick_df}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_auxiliary_variables(_bsrt_df):
    return BSRT_analysis.plot_auxiliary_variables({'show_plots': False,
                                                   'outputdir': None,
                                                   'kick_df': None}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_auxiliary_variables_with_kick_df(_bsrt_df, _kick_df):
    return BSRT_analysis.plot_auxiliary_variables({'show_plots': False,
                                                   'outputdir': None,
                                                   'kick_df': _kick_df}, _bsrt_df)


@pytest.mark.mpl_image_compare
def test_crossection_for_timesteps(_bsrt_df, _kick_df):
    results = BSRT_analysis.plot_crosssection_for_timesteps({'show_plots': False,
                                                             'outputdir': None,
                                                             'kick_df': _kick_df}, _bsrt_df)
    assert len(results) == len(_kick_df)
    return results[0]


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

@pytest.fixture()
def _kick_df():
    return pd.DataFrame(index=['2018_07_24@11_38_30_000000',
                               '2018_07_24@11_39_00_000000',
                               '2018_07_24@11_39_30_000000',
                               '2018_07_24@11_40_00_000000',
                               '2018_07_24@11_40_30_000000'])
                       