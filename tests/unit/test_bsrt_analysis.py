from ast import literal_eval
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pylhc import BSRT_analysis

# Forcing non-interactive Agg backend so rendering is done similarly across platforms during tests
matplotlib.use("Agg")

INPUTS_DIR = Path(__file__).parent.parent / "inputs"
BSRT_INPUTS = INPUTS_DIR / "bsrt_analysis"


def test_bsrt_df(_bsrt_df):
    results = BSRT_analysis.main(directory=str(BSRT_INPUTS), beam="B1")
    assert_frame_equal(results["bsrt_df"], _bsrt_df)


def test_select_by_time():
    time_df = pd.DataFrame(index=np.linspace(0, 10, 11), data={"A": np.linspace(0, 10, 11)})
    with pytest.raises(AssertionError):
        BSRT_analysis._select_files({"starttime": 3, "endtime": 1}, time_df)
    assert_frame_equal(
        BSRT_analysis._select_files({"starttime": 1, "endtime": 3}, time_df), time_df.loc[1:3]
    )
    assert_frame_equal(
        BSRT_analysis._select_files({"starttime": 1, "endtime": None}, time_df), time_df.loc[1:]
    )
    assert_frame_equal(
        BSRT_analysis._select_files({"starttime": None, "endtime": 3}, time_df), time_df.loc[:3]
    )


class TestPlotting:
    @pytest.mark.mpl_image_compare
    def test_fitvarplot(self, _bsrt_df):
        return BSRT_analysis.plot_fit_variables(
            {"show_plots": False, "outputdir": None, "kick_df": None}, _bsrt_df
        )

    @pytest.mark.mpl_image_compare
    def test_fitvarplot_with_kick_df(self, _bsrt_df, _kick_df):
        return BSRT_analysis.plot_fit_variables(
            {"show_plots": False, "outputdir": None, "kick_df": _kick_df}, _bsrt_df
        )

    @pytest.mark.mpl_image_compare
    def test_fullcrossection(self, _bsrt_df):
        return BSRT_analysis.plot_full_crosssection(
            {"show_plots": False, "outputdir": None, "kick_df": None}, _bsrt_df
        )

    @pytest.mark.mpl_image_compare
    def test_fullcrossection_with_kick_df(self, _bsrt_df, _kick_df):
        return BSRT_analysis.plot_full_crosssection(
            {"show_plots": False, "outputdir": None, "kick_df": _kick_df}, _bsrt_df
        )

    @pytest.mark.mpl_image_compare
    def test_auxiliary_variables(self, _bsrt_df):
        return BSRT_analysis.plot_auxiliary_variables(
            {"show_plots": False, "outputdir": None, "kick_df": None}, _bsrt_df
        )

    @pytest.mark.mpl_image_compare
    def test_auxiliary_variables_with_kick_df(self, _bsrt_df, _kick_df):
        return BSRT_analysis.plot_auxiliary_variables(
            {"show_plots": False, "outputdir": None, "kick_df": _kick_df}, _bsrt_df
        )

    @pytest.mark.mpl_image_compare
    def test_crossection_for_timesteps(self, _bsrt_df, _kick_df):
        results = BSRT_analysis.plot_crosssection_for_timesteps(
            {"show_plots": False, "outputdir": None, "kick_df": _kick_df}, _bsrt_df
        )
        assert len(results) == len(_kick_df)
        return results[0]


@pytest.fixture()
def _bsrt_df() -> pd.DataFrame:
    return pd.read_csv(
        BSRT_INPUTS / BSRT_analysis._get_bsrt_tfs_fname("B1"),
        parse_dates=True,
        index_col="TimeIndex",
        quotechar='"',
        converters={
            "acquiredImageRectangle": literal_eval,
            "beam": literal_eval,
            "gateMode": literal_eval,
            "imageSet": literal_eval,
            "lastFitResults": literal_eval,
            "projDataSet1": literal_eval,
            "projDataSet2": literal_eval,
            "projPositionSet1": literal_eval,
            "projPositionSet2": literal_eval,
        },
    )


@pytest.fixture()
def _kick_df() -> pd.DataFrame:
    return pd.DataFrame(
        index=[
            "2018_07_24@11_38_30_000000",
            "2018_07_24@11_39_00_000000",
            "2018_07_24@11_39_30_000000",
            "2018_07_24@11_40_00_000000",
            "2018_07_24@11_40_30_000000",
        ]
    )
