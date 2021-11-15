from typing import List, Iterable

import numpy as np
import pandas as pd
import pytest
import re
import tfs
from pathlib import Path
from pandas.testing import assert_frame_equal, assert_series_equal
from pylhc.irnl_rdt_correction import (
    main as irnl_correct, BETA, KEYWORD, X, Y, MULTIPOLE,
    get_integral_sign, list2str, switch_signs_for_beam4,
    IRCorrector
)
from pylhc.utils import tfs_tools

INPUTS = Path(__file__).parent.parent / "inputs"
LHC_MODELS_PATH = INPUTS / "model_lhc_thin_30cm"

EPS = 1e-13  # to compare floating point numbers against
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # the alphabet
MAX_N = 6  # 2 == Sextupole

PLACEHOLDER = "PLACEHOLDER"  # MADX Keyword PLACEHOLDER

# fields of IRCorrector --> columns in corrections tfs
VALUE = "value"
STRENGTH = "strength_component"
FIELD = "field_component"
ORDER = "order"
IP = "ip"
CIRCUIT = "circuit"
NAME = "name"


class TestStandardCorrection:
    @pytest.mark.parametrize('order', range(3, MAX_N+1))  # 3 == Sextupole
    @pytest.mark.parametrize('orientation', ('skew', 'normal'))
    @pytest.mark.parametrize('accel', ('lhc', 'hllhc'))
    def test_basic_correction(self, tmp_path: Path, order: int, orientation: str, accel: str):
        """Tests the basic correction functionality and performs some sanity checks.
        Operates on a pseudo-model so that the corrector values are easily known.
        Sanity Checks:
        - all correctors found
        - correctors have the correct value (as set by errors or zero)
        - all corrector circuits present in madx-script
        """
        # Parameters -----------------------------------------------------------
        if accel == 'lhc':
            if order == 5:
                pytest.skip("LHC has no decapole correctors")
            if order == 6 and orientation == 'skew':
                pytest.skip("LHC has no skew dodecapole correctors")

        orientation = "S" if orientation is "skew" else ""

        correct_ips = (1, 3)
        error_value = 2
        n_magnets = 4
        n_ips = 4

        n_correct_ips = len(correct_ips)
        n_sides = len("LR")
        n_orientation = len(["S", ""])

        # Setup ----------------------------------------------------------------
        optics = generate_pseudo_model(accel=accel, n_ips=n_ips, n_magnets=n_magnets)
        errors = generate_errortable(index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets))
        error_component = f"K{order-1}{orientation}L"
        errors[error_component] = error_value

        if order % 2:  # order is odd -> sides have different sign in rdt
            left_hand_magnets = errors.index.str.match(r".*L\d$")
            errors.loc[left_hand_magnets, error_component] = errors.loc[left_hand_magnets, error_component] / 2  # so they don't fully compensate

        # Correction -----------------------------------------------------------
        madx_corrections, df_corrections = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            output=tmp_path / "correct",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        # Testing --------------------------------------------------------------
        # Check output data ---
        assert len(list(tmp_path.glob("correct.*"))) == 2

        # Check all found correctors ---
        if accel == 'lhc':
            assert len(df_corrections.index) == (
                    n_orientation * n_sides * n_correct_ips * len("SO") +
                    n_sides * n_correct_ips * len("T")
            )

        if accel == 'hllhc':
            assert len(df_corrections.index) == n_orientation * n_sides * n_correct_ips * len("SODT")

        # All circuits in madx script ---
        for circuit in df_corrections[CIRCUIT]:
            assert circuit in madx_corrections

        # Check corrector values ---
        for test_order in range(3, MAX_N+1):
            for test_orientation in ("S", ""):
                for ip in correct_ips:
                    mask = (
                            (df_corrections[STRENGTH] == f"K{test_order-1}{test_orientation}L") &
                            (df_corrections[IP] == ip)
                            )
                    if (test_order == order) and (test_orientation == orientation):
                        if order % 2:
                            corrector_strengths = sum(df_corrections.loc[mask, VALUE])
                            assert abs(corrector_strengths) < EPS  # correctors should be equally distributed

                            corrector_strengths = -sum(df_corrections.loc[mask, VALUE].abs())
                            # as beta = 1
                            error_strengths = n_magnets * error_value / 2  # account for partial compensation (from above)
                        else:
                            corrector_strengths = sum(df_corrections.loc[mask, VALUE])
                            assert all(abs(df_corrections.loc[mask, VALUE] - corrector_strengths / n_sides) < EPS)
                            # as beta = 1
                            error_strengths = (n_sides * n_magnets * error_value)
                        assert abs(corrector_strengths + error_strengths) < EPS  # compensation of RDT
                    else:
                        assert all(df_corrections.loc[mask, VALUE] == 0.)

    @pytest.mark.parametrize('beam', (1, 2, 4))
    def test_lhc_correction(self, tmp_path: Path, beam: int):
        """Test LHC optics with random errors assigned.
        Sanity Checks:
        - all correctors found
        - all correctors have a value
        - all corrector circuits present in madx-script
        """
        # Setup ----------------------------------------------------------------
        np.random.seed(20211108)
        optics = read_lhc_model(beam)
        mask_ir = _get_ir_magnets_mask(optics.index)
        optics = optics.loc[mask_ir, :]
        correctors = optics.index[_get_corrector_magnets_mask(optics.index)]
        correct_ips = (1, 5)
        correctors = [c for c in correctors if int(c[-1]) in correct_ips]

        errors = generate_errortable(index=optics.index)

        # here: 2 == sextupole
        errors.loc[:, [f"K{order}{orientation}L"
                       for order in range(2, MAX_N) for orientation in ("S", "")]] = np.random.random([len(errors.index), 8])
        if beam == 4:
            negative_columns = _get_opposite_sign_beam4_kl_columns(range(2, MAX_N))
            errors.loc[:, negative_columns] = -errors.loc[:, negative_columns]

        # Correction -----------------------------------------------------------
        madx_corrections, df_corrections = irnl_correct(
            accel='lhc',
            optics=[optics],
            errors=[errors],
            beams=[beam],
            output=tmp_path / "correct",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        # Testing --------------------------------------------------------------
        # Check output data ---
        assert len(list(tmp_path.glob("correct.*"))) == 2

        # All correctors present with a value ---
        assert len(df_corrections.index) == 2 * 2 * 5 - 1  # sides * ips * corrector orders - faulty MCOSX.3L1
        assert all(df_corrections[VALUE] != 0)

        found_correctors = df_corrections[NAME].to_numpy()
        for name in correctors:
            if optics.loc[name, KEYWORD] == PLACEHOLDER:
                continue
            assert name in found_correctors

        # all corrector strengths are negative because all errors are positive (np.random.random)
        # this checks, that there is no sign-change between beam 1, 2 and 4.
        assert all(df_corrections[VALUE] < 0)

        # All circuits in madx script ---
        for circuit in df_corrections[CIRCUIT]:
            assert circuit in madx_corrections


class TestRDT:
    def test_different_rdts(self, tmp_path: Path):
        """Test that different RDTs can be corrected and only their correctors
        are returned. Also checks that the corrector values are varying between RDTs
        when they should. Octupole RDTs are used for this example.
        """
        # Parameters -----------------------------------------------------------
        accel = 'lhc'

        correct_ips = (1, 3)
        error_value = 2
        n_magnets = 4
        n_ips = 4

        # Setup ----------------------------------------------------------------
        optics = generate_pseudo_model(accel=accel, n_ips=n_ips, n_magnets=n_magnets)

        # use different beta for correctors to avoid beta cancellation
        # so that different RDTs give different corrector strengths
        correctors_mask = _get_corrector_magnets_mask(optics.index)
        optics.loc[correctors_mask, f"{BETA}Y"] = 3

        errors = generate_errortable(index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets))
        errors["K3L"] = error_value

        # Correction -----------------------------------------------------------
        _, df_corrections_f4000 = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=["f4000",],
            output=tmp_path / "correct4000",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        _, df_corrections_f2200 = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=["f2200",],
            output=tmp_path / "correct2200",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        _, df_corrections_f2002 = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=["f2002", ],
            output=tmp_path / "correct2002",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        # Testing --------------------------------------------------------------
        # Check output data ---
        assert len(list(tmp_path.glob("correct*"))) == 6

        # Check all found correctors ---
        # only octupole correctors should be present
        for correction in (df_corrections_f4000, df_corrections_f2200, df_corrections_f2002):
            assert len(correction.index) == 4
            assert all(correction['order'] == 4)

        # f4000 and f2200 should give same values for correction
        assert_frame_equal(df_corrections_f4000, df_corrections_f2200)

        # f4000 and f2002 should give different values for correction
        with pytest.raises(AssertionError):
            assert_series_equal(df_corrections_f4000[VALUE], df_corrections_f2002[VALUE])

        # frames are equal apart from value, though
        non_val_columns = [col for col in df_corrections_f2200.columns if col != VALUE]
        assert_frame_equal(df_corrections_f4000[non_val_columns], df_corrections_f2002[non_val_columns])


class TestFeeddown:
    @pytest.mark.parametrize('x', (2, 0))
    @pytest.mark.parametrize('y', (1.5, 0))
    def test_general_feeddown(self, tmp_path: Path, x: float, y: float):
        """Test feeddown functionality from decapoles to octupoles and sextupoles."""
        # Parameters -----------------------------------------------------------
        accel = 'lhc'

        correct_ips = (1, 3)
        error_value = 2
        n_magnets = 4
        n_ips = 4
        n_sides = 2

        # Setup ----------------------------------------------------------------
        optics = generate_pseudo_model(
            accel=accel, n_ips=n_ips, n_magnets=n_magnets, x=x, y=y)
        errors = generate_errortable(
            index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        )
        errors["K4L"] = error_value  # normal decapole errors

        # Correction ---------------------------------------------------------------
        rdts = "f4000", f"f3001"
        _, df_corrections = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=rdts,
            output=tmp_path / "correct",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        _, df_corrections_fd1 = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=rdts,
            output=tmp_path / "correct_fd1",
            feeddown=1,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        errors["K4L"] = 0
        errors["K5L"] = error_value  # normal dodecapole errors
        _, df_corrections_fd2 = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=rdts,
            output=tmp_path / "correct_fd2",
            feeddown=2,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        # Testing ------------------------------------------------------------------
        # Check output data ---
        assert len(list(tmp_path.glob("correct*"))) == 6

        # Check all found correctors ---
        # no corrections with feed-down
        assert all(df_corrections[VALUE] == 0)

        if x == 0 and y == 0:
            assert all(df_corrections_fd1[VALUE] == 0)
            assert all(df_corrections_fd2[VALUE] == 0)

        else:
            for ip in correct_ips:
                normal_oct_mask = (df_corrections[STRENGTH] == "K3L") & (df_corrections[IP] == ip)
                skew_oct_mask = (df_corrections[STRENGTH] == "K3SL") & (df_corrections[IP] == ip)
                dodecapole_error_sum = error_value * n_magnets * n_sides
                norm_oct_corr_fd1 = sum(df_corrections_fd1.loc[normal_oct_mask, VALUE])
                skew_oct_corr_fd1 = sum(df_corrections_fd1.loc[skew_oct_mask, VALUE])
                assert abs(norm_oct_corr_fd1 + x * dodecapole_error_sum) < EPS
                assert abs(skew_oct_corr_fd1 + y * dodecapole_error_sum) < EPS

                norm_oct_corr_fd2 = sum(df_corrections_fd2.loc[normal_oct_mask, VALUE])
                skew_oct_corr_fd2 = sum(df_corrections_fd2.loc[skew_oct_mask, VALUE])
                assert abs(norm_oct_corr_fd2 + 0.5 * (x**2 - y**2) * dodecapole_error_sum) < EPS
                assert abs(skew_oct_corr_fd2 + x * y * dodecapole_error_sum) < EPS


    @pytest.mark.parametrize('corrector', ("a5", "b5", "a6", "b6"))
    @pytest.mark.parametrize('x', (2, 0))
    @pytest.mark.parametrize('y', (2, 1.5, 0))
    def test_correct_via_feeddown(self, tmp_path: Path, x: float, y: float, corrector: str):
        """Test correct RDT via feeddown from higher order corrector.
        In this example: Use normal and skew deca- and dodecapole correctors
        to correct for normal octupole errors (which make it easy to
        just sum up over both sides).
        """
        # Parameters -----------------------------------------------------------
        accel = 'hllhc'

        correct_ips = (1, 3)
        error_value = 2
        n_magnets = 4
        n_ips = 4
        n_sides = 2

        # Setup ----------------------------------------------------------------
        optics = generate_pseudo_model(
            accel=accel, n_ips=n_ips, n_magnets=n_magnets, x=x, y=y)
        errors = generate_errortable(
            index=get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets),
        )
        errors["K3L"] = error_value  # octupole errors

        # Correction ---------------------------------------------------------------
        rdts = {"f4000": [corrector]}
        _, df_corrections = irnl_correct(
            accel=accel,
            optics=[optics],
            errors=[errors],
            beams=[1],
            rdts=rdts,
            output=tmp_path / "correct",
            feeddown=0,
            ips=correct_ips,
            ignore_missing_columns=True,
            iterations=1,
        )

        assert len(df_corrections.index) == len(correct_ips) * n_sides
        assert all(df_corrections[FIELD] == corrector)

        coeff = {"a5": y, "b5": x, "a6": y*x, "b6": 0.5*(x**2 - y**2)}[corrector]
        if coeff == 0:
            # No Feed-down possible
            assert all(df_corrections[VALUE] < EPS)
        else:
            # as beta = 1
            error_strengths = (n_sides * n_magnets * error_value)
            for ip in correct_ips:
                mask = df_corrections[IP] == ip
                corrector_strengths = coeff * sum(df_corrections.loc[mask, VALUE])
                assert abs(corrector_strengths + error_strengths) < EPS  # compensation of RDT


class TestUnit:
    """Unit Tests for easy to test functions (sorry for not testing more)."""
    def test_get_integral_sign(self):
        for n in range(10):
            assert get_integral_sign(n, "R") == (-1)**n
            assert get_integral_sign(n, "L") == 1

    def test_list_to_str(self):
        assert ABC == "".join(list2str(list(ABC)).replace(" ", "").replace("'", "").replace('"', "").split(','))

    def test_wrong_arguments(self):
        with pytest.raises(AttributeError) as e:
            irnl_correct(
                feddown=0,
                itterations=1,
            )
        assert "feddown" in str(e)
        assert "itterations" in str(e)

    @pytest.mark.parametrize('beam', (1, 2, 4))
    def test_switch_signs(self, beam: int):
        all_k = [f"K{order}{orientation}L" for order in range(2, MAX_N) for orientation in ("S", "")]
        optics = generate_pseudo_model(n_ips=1, n_magnets=10, accel='lhc', x=10, y=5)
        optics[all_k] = 1

        errors = generate_errortable(index=optics.index, value=2.)

        # make copies as it switches in place
        optics_switch = optics.copy()
        errors_switch = errors.copy()
        switch_signs_for_beam4([optics_switch], [errors_switch], [beam])

        if beam != 4:
            assert_frame_equal(optics, optics_switch)
            assert_frame_equal(errors, errors_switch)
        else:
            # in madx optics only X changes sign for beam 4 ...
            switch_col_optics_mask = optics.columns.isin(["X"])
            assert_frame_equal(optics.loc[:, switch_col_optics_mask], -optics_switch.loc[:, switch_col_optics_mask])
            assert_frame_equal(optics.loc[:, ~switch_col_optics_mask], optics_switch.loc[:, ~switch_col_optics_mask])

            # ... but in the errors DX and the anti-symmetric KL change sign
            switch_col_errors_mask = errors.columns.isin(["DX"] + _get_opposite_sign_beam4_kl_columns(range(MAX_N)))
            assert_frame_equal(errors.loc[:, switch_col_errors_mask], -errors_switch.loc[:, switch_col_errors_mask])
            assert_frame_equal(errors.loc[:, ~switch_col_errors_mask], errors_switch.loc[:, ~switch_col_errors_mask])


# Helper -------------------------------------------------------------------------------------------

def read_lhc_model(beam: int) -> tfs.TfsDataFrame:
    """Read the LHC model from the input directory."""
    # tfs files were too big, but if generated from the `.madx` the `.tfs` can be used directly.
    # E.g. for debugging purposes.
    # return tfs.read_tfs(LHC_MODELS_PATH / f"twiss.lhc.b{beam}.nominal.tfs", index="NAME")
    return tfs_tools.read_hdf(LHC_MODELS_PATH / f"twiss.lhc.b{beam}.nominal.hd5")


def generate_pseudo_model(n_ips: int, n_magnets: int, accel: str,
                          betax: float = 1, betay: float = 1, x: float = 0, y: float = 0) -> pd.DataFrame:
    """Generate a Twiss-Like DataFrame with magnets as index and Beta and Orbit columns."""
    df = pd.DataFrame(
        index=(
            get_some_magnet_names(n_ips=n_ips, n_magnets=n_magnets) +
            get_lhc_corrector_names(n_ips=n_ips, accelerator=accel)
        ),
        columns=[f"{BETA}{X}", f"{BETA}{Y}", X, Y, KEYWORD]
    )
    df[f"{BETA}{X}"] = betax
    df[f"{BETA}{Y}"] = betay
    df[X] = x
    df[Y] = y
    df[KEYWORD] = MULTIPOLE
    return df


def generate_errortable(index: pd.Series, value: float = 0) -> pd.DataFrame:
    """Return DataFrame from index and KN(S)L + D[XY] columns."""
    return pd.DataFrame(value,
                        index=index,
                        columns=[f"K{n}{o}L" for n in range(MAX_N) for o in ("", "S")] + [f"D{plane}" for plane in "XY"]
                        )


def get_some_magnet_names(n_ips: int, n_magnets: int) -> List[str]:
    r"""More or less random magnet names, ending in ``[LR]\d``.
    n_magnets < 26 because their names come from alphabet.
    """
    return [
        f"M{name}.{number+1}{side}{ip}"
        for ip in range(1, n_ips+1)
        for side in "LR"
        for number, name in enumerate(ABC[:n_magnets])
    ]


def get_lhc_corrector_names(n_ips: int, accelerator: str = 'lhc') -> List[str]:
    r"""Corrector names as defined in LHC/HLLHC as the correction script is looking for them.
    Need to start with ``MC`` and end in ``X.3[LR]\d`` or ``XF.3[LR][15]``"""
    magnets = [
        f"MC{order}{orientation}X.3{side}{ip}"
        for order in "SODT"
        for orientation in ("S", "")
        for side in "LR"
        for ip in range(1, n_ips+1)
    ]
    if accelerator == 'hllhc':
        magnets = [
            name.replace("X", "XF") if name[-1] in "15" else name
            for name in magnets
        ]
    return magnets


def _get_ir_magnets_mask(index: pd.Index) -> pd.Series:
    """Returns a boolean mask for magnets in the IR (n<=13) in the index."""
    return index.str.match(r"M.*\.(1[0123]|[0-9])[LR]\d(\.B\d)?", flags=re.IGNORECASE)


def _get_corrector_magnets_mask(index: pd.Index) -> pd.Series:
    """Returns a boolean mask for the nonlinear corrector magnets in index."""
    return index.str.match(r"MC.*XF?\.3[LR]\d$", flags=re.IGNORECASE)


def _get_opposite_sign_beam4_kl_columns(range_: Iterable):
    """Get the KN(S)L columns that have opposite signs for beam 4."""
    return [f"K{order}{'' if order % 2 else 'S'}L" for order in range_]
