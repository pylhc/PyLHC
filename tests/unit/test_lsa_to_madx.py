import json

from pathlib import Path
from typing import Dict
from webbrowser import get

import pytest
import tfs

from pandas._testing import assert_dict_equal

from pylhc.lsa_to_madx import (
    _get_trim_variable,
    get_madx_script_from_definition_dataframe,
    parse_knobs_and_trim_values_from_file,
)

INPUTS_DIR = Path(__file__).parent.parent / "inputs"
LSA_TO_MADX_INPUTS = INPUTS_DIR / "lsa_to_madx"


class TestParsing:
    def test_parse_knob_definition_file(self, knobs_file, parsed_definitions):
        solution = parse_knobs_and_trim_values_from_file(knobs_file=knobs_file)
        assert_dict_equal(solution, parsed_definitions)


class TestMADXWriting:
    def test_madx_script_writing_from_definition_df(self, knob_definition_df, correct_madx_script):
        script = get_madx_script_from_definition_dataframe(knob_definition_df, lsa_knob="LHCBEAM/ATS_Test_Knob")
        assert script == correct_madx_script

    @pytest.mark.parametrize("lsa_knob", ["LHCBEAM/Super_Duper_Long_Name_For_A_Knob_Will_Be_Truncated_For_Sure", "ATS_Test_Knob"])
    def test_trim_variable_from_long_knob_name(self, lsa_knob):
        """Testing that the generated trim variable is correctly truncated if too long."""
        assert (
            _get_trim_variable("ATS_2022_05_08_B1_arc_by_arc_coupling_133cm_30cm")
            == "trim_22_05_08_B1_arc_by_arc_coupling_133cm_30cm"
        )
        assert _get_trim_variable("___knob") == "trim_knob"  # make sure we handle several underscores
        assert len(_get_trim_variable(lsa_knob)) < 48


# ----- Fixtures ----- #


@pytest.fixture()
def knobs_file() -> Path:
    """A test file with various knobs from a 2017 optics"""
    return LSA_TO_MADX_INPUTS / "knobs_definitions.txt"


@pytest.fixture()
def knob_definition_df() -> tfs.TfsDataFrame:
    """
    TfsDataFrame returned by
    LSA.get_knob_circuits(
        optics="R2022a_A30cmC30cmA10mL200cm",
        knob_name="LHCBEAM/MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos"
    )
    """
    return tfs.read(LSA_TO_MADX_INPUTS / "MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos.tfs")


@pytest.fixture()
def parsed_definitions() -> Dict[str, float]:
    with (LSA_TO_MADX_INPUTS / "parsed_definitions.json").open("r") as f:
        defs = json.load(f)
    return defs


@pytest.fixture
def correct_madx_script() -> str:
    """Script for LHCBEAM/MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos_knob with trim at +1"""
    return (LSA_TO_MADX_INPUTS / "LHCBEAM_MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos_knob.madx").read_text()
