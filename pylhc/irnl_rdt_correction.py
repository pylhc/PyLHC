"""
Nonlinear Correction calculation in the IRs
--------------------------------------------

Performs local correction of the Resonance Driving Terms (RDTs)
in the Insertion Regions (IRs) based on the principle described in
[#BruningDynamicApertureStudies2004]_ with the addition of correcting
feed-down and using feed-down to correct lower order RDTs.
Details can be found in [#DillyNonlinearIRCorrections2022]_ .

This script is written to be used stand-alone if needed
with python 3.6+ and with ``tfs-pandas`` installed.


.. rubric:: References

..  [#BruningDynamicApertureStudies2004]
    O. Bruening et al.,
    Dynamic aperture studies for the LHC separation dipoles. (2004)
    https://cds.cern.ch/record/742967

..  [#DillyNonlinearIRCorrections2022]
    J. Dilly et al.,
    Corrections of high-order nonlinear errors in the LHC and HL-LHC insertion regions. (2022)


author: Joschua Dilly

"""
import argparse
import logging
import sys
from pathlib import Path
from time import time
from typing import Sequence, Tuple, Iterable, Sized, Set, Callable

import numpy as np
import tfs
from pandas import DataFrame, Series

LOG = logging.getLogger(__name__)


# Classes ----------------------------------------------------------------------
class IRCorrector:
    def __init__(self, field_component: str, accel: str, ip: int, side: str):
        order, skew = field_component2order(field_component)

        self.field_component = field_component
        self.order = order
        self.skew = skew
        self.accel = accel
        self.ip = ip
        self.side = side

        main_name = f'C{ORDER_NAME_MAP[order]}{SKEW_NAME_MAP[skew]}X'
        extra = "F" if accel == "hllhc" and ip in [1, 5] else ""
        self.type = f'M{main_name}{extra}'
        self.name = f'{self.type}.{POSITION:d}{side}{ip:d}'
        self.circuit = f'K{main_name}{POSITION:d}.{side}{ip:d}'
        self.strength_component = f"K{order-1}{SKEW_NAME_MAP[skew]}L"  # MAD-X order notation
        self.value = 0

    def __repr__(self):
        return f"IRCorrector object {str(self)}"

    def __str__(self):
        return f"{self.name} ({self.field_component}), {self.strength_component}: {self.value: 6E}"

    def __lt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                if self.ip == other.ip:
                    return (self.side == SIDES[1]) < (other.side == SIDES[1])
                return self.ip < other.ip
            return self.skew < other.skew
        return self.order < other.order

    def __gt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                if self.ip == other.ip:
                    return (self.side == SIDES[1]) > (other.side == SIDES[1])
                return self.ip > other.ip
            return self.skew > other.skew
        return self.order > other.order

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class RDT:
    def __init__(self, name: str):
        self.name = name
        self.jklm = tuple(int(i) for i in name[1:5])
        self.j, self.k, self.l, self.m = self.jklm
        self.skew = bool((self.l + self.m) % 2)
        self.order = sum(self.jklm)
        self.swap_beta_exp = name.endswith("*")  # swap beta-exponents

    def __repr__(self):
        return f"RDT object {str(self)}"

    def __str__(self):
        return f"{self.name} ({order2field_component(self.order, self.skew)})"

    def __lt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                return len(self.name) < len(other.name)
            return self.skew < other.skew
        return self.order < other.order

    def __gt__(self, other):
        if self.order == other.order:
            if self.skew == other.skew:
                return len(self.name) > len(other.name)
            return self.skew > other.skew
        return self.order > other.order

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


# Additional Classes ---

class DotDict(dict):
    """ Make dict fields accessible by attributes."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        """ Needed to raise the correct exceptions """
        try:
            return super(DotDict, self).__getitem__(key)
        except KeyError as e:
            raise AttributeError(e).with_traceback(e.__traceback__) from e


class Timer:
    """ Collect Times and print a summary at the end. """
    def __init__(self, name: str = "start", print_fun: Callable[[str], None] = print):
        self.steps = {}
        self.step(name)
        self.print = print_fun

    def step(self, name: str = None):
        if not name:
            name = str(len(self.steps))
        time_ = time()
        self.steps[name] = time_
        return time_

    def time_since_step(self, step=None):
        if not step:
            step = list(self.steps.keys())[-1]
        dtime = time() - self.steps[step]
        return dtime

    def time_between_steps(self, start: str = None, end: str = None):
        list_steps = list(self.steps.keys())
        if not start:
            start = list_steps[0]
        if not end:
            end = list_steps[-1]
        dtime = self.steps[end] - self.steps[start]
        return dtime

    def summary(self):
        str_length = max(len(s) for s in self.steps.keys())
        time_length = len(f"{int(self.time_between_steps()):d}")
        format_str = (f"{{step:{str_length}s}}:"
                      f" +{{dtime: {time_length:d}.5f}}s"
                      f" ({{ttime: {time_length:d}.3f}}s total)")
        last_time = None
        start_time = None
        for step, step_time in self.steps.items():
            if last_time is None:
                start_time = step_time
                self.print(f"Timing Summary ----")
                self.print(format_str.format(step=step, dtime=0, ttime=0))
            else:
                self.print(format_str.format(
                    step=step, dtime=step_time-last_time, ttime=step_time-start_time)
                )
            last_time = step_time


# Constants --------------------------------------------------------------------

POSITION = 3  # all NL correctors are at POSITION 3, to be adapted for Linear
ORDER_NAME_MAP = {1: "B", 2: "Q", 3: "S", 4: "O", 5: "D", 6: "T"}
SKEW_NAME_MAP = {True: "S", False: ""}
SKEW_FIELD_MAP = {True: "a", False: "b"}
FIELD_SKEW_MAP = {v: k for k, v in SKEW_FIELD_MAP.items()}
SKEW_CHAR_MAP = {True: "J", False: "K"}

PLANES = ("X", "Y")
BETA = "BET"
DELTA = "D"
KEYWORD = "KEYWORD"
MULTIPOLE = "MULTIPOLE"
X, Y = PLANES
SIDES = ("L", "R")


# Default input options
DEFAULTS = {'feeddown': 0,
            'ips': [1, 2, 5, 8],
            'accel': 'lhc',
            'solver': 'lstsq',
            'update_optics': True,
            'iterations': 1,
            'ignore_corrector_settings': False,
            'rdts2': None,
            'ignore_missing_columns': False,
            'output': None,
            }

DEFAULT_RDTS = {
    'lhc': ('F0003', 'F0003*',  # correct a3 errors with F0003
            'F1002', 'F1002*',  # correct b3 errors with F1002
            'F1003', 'F3001',  # correct a4 errors with F1003 and F3001
            'F4000', 'F0004',  # correct b4 errors with F4000 and F0004
            'F6000', 'F0006',  # correct b6 errors with F6000 and F0006
            ),
    'hllhc': ('F0003', 'F0003*',  # correct a3 errors with F0003
              'F1002', 'F1002*',  # correct b3 errors with F1002
              'F1003', 'F3001',  # correct a4 errors with F1003 and F3001
              'F0004', 'F4000',  # correct b4 errors with F0004 and F4000
              'F0005', 'F0005*',  # correct a5 errors with F0005
              'F5000', 'F5000*',  # correct b5 errors with F5000
              'F5001', 'F1005',  # correct a6 errors with F5001 and F1005
              'F6000', 'F0006',  # correct b6 errors with F6000 and F0006
              ),
}

EXT_TFS = ".tfs"  # suffix for dataframe file
EXT_MADX = ".madx"  # suffix for madx-command file


# Solving functions ---

def _solve_linear(correctors, lhs, rhs):
    res = np.linalg.solve(lhs, rhs)
    _assign_corrector_values(correctors, res)


def _solve_invert(correctors, lhs, rhs):
    res = np.linalg.inv(lhs).dot(rhs)
    _assign_corrector_values(correctors, res)


def _solve_lstsq(correctors, lhs, rhs):
    res = np.linalg.lstsq(lhs, rhs, rcond=None)
    _assign_corrector_values(correctors, res[0])
    if len(res[1]):
        LOG.info(f"Residuals ||I - Bx||_2: {list2str(res[1])}")
    LOG.debug(f"Rank of Beta-Matrix: {res[2]}")


SOLVER_MAP = {'inv': _solve_invert,
              'lstsq': _solve_lstsq,
              'linear': _solve_linear,}


APPROXIMATE_SOLVERS = ['lstsq']


# Main Functions ---

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optics",
        dest="optics",
        nargs="+",
        help="Path(s) to optics file(s). Defines which elements to correct for!",
        required=True,
    )
    parser.add_argument(
        "--errors",
        dest="errors",
        nargs="+",
        help="Path(s) to error file(s).",
        required=True,
    )
    parser.add_argument(
        "--beams",
        dest="beams",
        type=int,
        nargs="+",
        help="Which beam the files come from (1, 2 or 4)",
        required=True,
    )
    parser.add_argument(
        "--output",
        dest="output",
        help=("Path to write command and tfs_df file. "
              f"Extension (if given) is ignored and replaced with {EXT_TFS} and {EXT_MADX} "
              "for the Dataframe and the command file respectively."
              ),
    )
    parser.add_argument(
        "--rdts",
        dest="rdts",
        nargs="+",
        help=("RDTs to correct."
              " Format: 'Fjklm'; or 'Fjklm*'"
              " to correct for this RDT in Beam 2 using"
              " beta-symmetry (jk <-> lm)."),
    )
    parser.add_argument(
        "--rdts2",
        dest="rdts2",
        nargs="+",
        help=("RDTs to correct for second beam/file, if different from first."
              " Same format rules as for `rdts`."),
    )
    parser.add_argument(
        "--accel",
        dest="accel",
        type=str.lower,
        choices=list(DEFAULT_RDTS.keys()),
        default=DEFAULTS['accel'],
        help="Which accelerator we have.",
    )
    parser.add_argument(
        "--feeddown",
        dest="feeddown",
        type=int,
        help="Order of Feeddown to include.",
        default=DEFAULTS['feeddown'],
    )
    parser.add_argument(
        "--ips",
        dest="ips",
        nargs="+",
        help="In which IPs to correct.",
        type=int,
        default=DEFAULTS['ips'],
    )
    parser.add_argument(
        "--solver",
        dest="solver",
        help="Solving method to use.",
        type=str.lower,
        choices=list(SOLVER_MAP.keys()),
        default=DEFAULTS['solver'],
    )
    parser.add_argument(
        "--update_optics",
        dest="update_optics",
        type=bool,
        help=("Sorts the RDTs to start with the highest order and updates the "
              "corrector strengths in the optics after calculation, so the "
              "feeddown to lower order correctors is included."
              ),
        default=DEFAULTS["update_optics"]
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        help=("Reiterate correction, "
              "starting with the previously calculated values."),
        default=DEFAULTS["iterations"]
    )
    parser.add_argument(
        "--ignore_corrector_settings",
        dest="ignore_corrector_settings",
        help=("Ignore the current settings of the correctors. If this is not set "
              "the corrector values of the optics are used as initial conditions."),
        action="store_true",
    )
    parser.add_argument(
        "--ignore_missing_columns",
        dest="ignore_missing_columns",
        help=("If set, missing strength columns in any of the input files "
              "are assumed to be zero, instead of raising an error."),
        action="store_true",
    )
    return parser


def main(**opt) -> Tuple[str, tfs.TfsDataFrame]:
    """ Get correctors and their optimal powering to minimize the given RDTs.

    Keyword Args:

        optics (list[str/Path/DataFrame]): Path(s) to optics file(s) or DataFrame(s) of optics.
                                           Needs to contain only the elements to be corrected
                                           (e.g. only the ones in the IRs).
                                           All elements from the error-files need to be present.
                                           Required!
        errors (list[str/Path/DataFrame]): Path(s) to error file(s) or DataFrame(s) of errors.
                                           Can contain less elements than the optics files,
                                           these elements are then assumed to have no errors.
                                           Required!
        beams (list[int]): Which beam the files come from (1, 2 or 4).
                           Required!
        output (str/Path): Path to write command and tfs_df file.
                           Extension (if given) is ignored and replaced with '.tfs' and '.madx'
                           for the Dataframe and the command file respectively.
                           Default: ``None``.
        rdts (list[str], dict[str, list[str]):
                          RDTs to correct.
                          Format: 'Fjklm'; or 'Fjklm*' to correct for
                          this RDT in Beam 2 using beta-symmetry (jk <-> lm).
                          The RDTs can be either given as a list, then the appropriate correctors
                          are determined by jklmn.
                          Alternatively, the input can be a dictionary,
                          where the keys are the RDT names as before, and the values are a list
                          of correctors to use, e.g. 'b5' for normal decapole corrector,
                          'a3' for skew sextupole, etc.
                          If the order of the corrector is higher than the order of the RDT,
                          the feed-down from the corrector is used for correction.
                          In the case where multiple orders of correctors are used,
                          increasing ``iterations`` might be useful.
                          Default: Standard RDTs for given ``accel`` (see ``DEFAULT_RDTS`` in this file).
        rdts2 (list[str], dict[str, list[str]):
                           RDTs to correct for second beam/file, if different from first.
                           Same format rules as for ``rdts``. Default: ``None``.
        accel (str): Which accelerator we have. One of 'lhc', 'hllhc'.
                     Default: ``lhc``.
        feeddown (int): Order of Feeddown to include.
                        Default: ``0``.
        ips (list[int]): In which IPs to correct.
                         Default: ``[1, 2, 5, 8]``.
        solver (str): Solver to use. One of 'lstsq', 'inv' or 'linear'.
                      Default ``lstsq``.
        update_optics (bool): Sorts the RDTs to start with the highest order
                              and updates the corrector strengths in the optics
                              after calculation, so the feeddown to lower order
                              correctors is included.
                              Default: ``True``.
        ignore_corrector_settings (bool): Ignore the current settings of the correctors.
                                          If this is not set the corrector values of the
                                          optics are used as initial conditions.
                                          Default: ``False``.
        ignore_missing_columns (bool): If set, missing strength columns in any
                                       of the input files are assumed to be
                                       zero, instead of raising an error.
                                       Default: ``False``.
        iterations (int): Reiterate correction, starting with the previously
                          calculated values.
                          Default: ``1``.


    Returns:

        tuple[string, Dataframe]:
        the string contains the madx-commands used to power the correctors;
        the dataframe contains the same values in a pandas DataFrame.
    """
    LOG.info("Starting IRNL Correction.")
    timer = Timer("Start", print_fun=LOG.debug)
    if not len(opt):
        opt = vars(get_parser().parse_args())
    opt = DotDict(opt)
    opt = _check_opt(opt)
    timer.step("Opt Parsed")

    rdt_maps = sort_rdts(opt.rdts, opt.rdts2)
    _check_corrector_order(rdt_maps, opt.update_optics, opt.feeddown)
    needed_orders = _get_needed_orders(rdt_maps, opt.feeddown)
    timer.step("RDT Sorted")

    optics_dfs = get_tfs(opt.optics)
    errors_dfs = get_tfs(opt.errors)
    timer.step("Optics Loaded")

    _check_dfs(optics_dfs, errors_dfs, opt.beams, needed_orders, opt.ignore_missing_columns)
    switch_signs_for_beam4(optics_dfs, errors_dfs, opt.beams)
    timer.step("Optics Checked")

    correctors = solve(rdt_maps, optics_dfs, errors_dfs, opt)
    timer.step("Done")

    timer.summary()
    if len(correctors) == 0:
        raise EnvironmentError('No correctors found in input optics.')

    return get_and_write_output(opt.output, correctors)


def solve(rdt_maps, optics_dfs, errors_dfs, opt):
    """ Calculate corrections.
    They are grouped into rdt's with common correctors. If possible, these are
    ordered from highest order to lowest, to be able to update optics and include
    their feed-down. """
    all_correctors = []
    remaining_rdt_maps = rdt_maps
    while remaining_rdt_maps:
        current_rdt_maps, remaining_rdt_maps, corrector_names = _get_current_rdt_maps(remaining_rdt_maps)

        for ip in opt.ips:
            correctors = get_available_correctors(corrector_names, opt.accel, ip, optics_dfs)
            all_correctors += correctors
            if not len(correctors):
                continue  # warnings are logged in get_available_correctors

            saved_corrector_values = init_corrector_and_optics_values(correctors, optics_dfs,
                                                                      opt.update_optics,
                                                                      opt.ignore_corrector_settings)

            beta_matrix, integral = build_equation_system(
                current_rdt_maps, correctors,
                ip, optics_dfs, errors_dfs, opt.feeddown,
            )
            for iteration in range(opt.iterations):
                solve_equation_system(correctors, beta_matrix, integral, opt.solver)  # changes corrector values
                update_optics(correctors, optics_dfs)  # update corrector values in optics

                # update integral values after iteration:
                integral_before = integral
                _, integral = build_equation_system(
                    current_rdt_maps, [],  # empty correctors list skips beta-matrix calculation
                    ip, optics_dfs, errors_dfs, opt.feeddown
                )
                _log_correction(integral_before, integral, current_rdt_maps, optics_dfs, iteration, ip)

            LOG.info(f"Correction of IP{ip:d} complete.")
            restore_optics_values(saved_corrector_values, optics_dfs)  # hint: nothing saved if update_optics is True
    return sorted(all_correctors)


def _get_current_rdt_maps(rdt_maps):
    """ Creates a new rdt_map with all rdt's that share correctors.  """
    n_maps = len(rdt_maps)
    new_rdt_map = [{} for _ in rdt_maps]
    for rdt_map in rdt_maps:
        # get next RDT/correctors
        if len(rdt_map):
            rdt, correctors = list(rdt_map.items())[0]
            break
    else:
        raise ValueError("rdt_maps are empty. "
                         "This should have triggered an end of the solver loop "
                         "earlier. Please investigate.")

    correctors = set(correctors)
    checked_correctors = set()
    while len(correctors):
        # find all RDTs with the same correctors
        checked_correctors |= correctors
        additional_correctors = set()  # new correctors found this loop

        for corrector in correctors:
            for idx in range(n_maps):
                for rdt_current, rdt_correctors in rdt_maps[idx].items():
                    if corrector in rdt_correctors:
                        new_rdt_map[idx][rdt_current] = rdt_correctors
                        additional_correctors |= (set(rdt_correctors) - checked_correctors)

        correctors = additional_correctors

    remaining_rdt_map = [{k: v for k, v in rdt_maps[idx].items()
                          if k not in new_rdt_map[idx].keys()} for idx in range(n_maps)]

    if not any(len(rrm) for rrm in remaining_rdt_map):
        remaining_rdt_map = None

    return new_rdt_map, remaining_rdt_map, checked_correctors


# RDT Sorting ------------------------------------------------------------------

def sort_rdts(rdts: Sequence, rdts2: Sequence) -> Tuple[dict, dict]:
    """ Sorts RDTs by reversed-order and orientation (skew, normal). """
    LOG.debug("Sorting RDTs")
    LOG.debug(" - First Optics:")
    rdt_dict = _build_rdt_dict(rdts)
    if rdts2 is not None:
        LOG.debug(" - Second Optics:")
        rdt_dict2 = _build_rdt_dict(rdts2)
    else:
        LOG.debug(" - Second Optics: same RDTs as first.")
        rdt_dict2 = rdt_dict.copy()
    return rdt_dict, rdt_dict2


def _build_rdt_dict(rdts: Sequence) -> dict:
    LOG.debug("Building RDT dictionary.")
    if not isinstance(rdts, dict):
        rdts = {rdt: [] for rdt in rdts}

    rdt_dict = {}
    for rdt_name, correctors in rdts.items():
        rdt = RDT(rdt_name)
        if not len(correctors):
            skew = rdt.skew
            correctors = [order2field_component(rdt.order, skew)]

        rdt_dict[rdt] = correctors
        LOG.debug(f"Added: {rdt} with correctors: {list2str(correctors)}")
    rdt_dict = dict(sorted(rdt_dict.items(), reverse=True))  # sorts by highest order and skew first
    return rdt_dict


def _get_needed_orders(rdt_maps: Sequence[dict], feed_down: int) -> Sequence[int]:
    """Returns the sorted orders needed for correction, based on the order
    of the RDTs to correct plus the feed-down involved and the order of the
    corrector, which can be higher than the RDTs in case one wants to correct
    via feeddown."""
    needed_orders = set()
    for rdt_map in rdt_maps:
        for rdt, correctors in rdt_map.items():
            # get orders from RDTs + feed-down
            for fd in range(feed_down+1):
                needed_orders |= {rdt.order + fd, }

            # get orders from correctors
            for corrector in correctors:
                needed_orders |= {int(corrector[1]), }
    return sorted(needed_orders)


# Equation System -------------------------------------------------------------

def build_equation_system(rdt_maps: Sequence[dict], correctors: Sequence[IRCorrector], ip: int,
                          optics_dfs: Sequence, errors_dfs: Sequence,
                          feeddown: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Builds equation system as in Eq. (43) in [#DillyNonlinearIRCorrections2022]_
    for a given ip for all given optics and error files (i.e. beams) and rdts.

    Returns
        b_matrix: np.array N_rdts x  N_correctors
        integral: np.array N_rdts x 1
     """
    n_rdts = sum(len(rdt_map.keys()) for rdt_map, _ in zip(rdt_maps, optics_dfs))
    b_matrix = np.zeros([n_rdts, len(correctors)])
    integral = np.zeros([n_rdts, 1])

    idx_row = 0  # row in equation system
    for idx_file, rdts, optics_df, errors_df in zip(range(1, 3), rdt_maps, optics_dfs, errors_dfs):
        for rdt, rdt_correctors in rdts.items():
            LOG.info(f"Calculating {rdt}, optics {idx_file}/{len(optics_dfs)}, IP{ip:d}")
            integral[idx_row] = get_elements_integral(rdt, ip, optics_df, errors_df, feeddown)

            for idx_corrector, corrector in enumerate(correctors):
                if corrector.field_component not in rdt_correctors:
                    continue
                b_matrix[idx_row][idx_corrector] = get_corrector_coefficient(rdt, corrector, optics_df, errors_df)
            idx_row += 1

    return b_matrix, integral


def _log_correction(integral_before: np.array, integral_after: np.array, rdt_maps: Sequence[dict],
                    optics_dfs: Sequence[DataFrame], iteration: int, ip: int):
    """ Log the correction initial and final value of this iteration. """
    LOG.info(f"RDT change in IP{ip:d}, iteration {iteration+1:d}:")
    integral_iter = zip(integral_before, integral_after)
    for idx_optics, rdts, _ in zip(range(1, 3), rdt_maps, optics_dfs):
        for rdt in rdts.keys():
            val_before, val_after = next(integral_iter)
            delta = val_after - val_before
            LOG.info(f"Optics {idx_optics}, {rdt.name}: {val_before[0]:.2e} -> {val_after[0]:.2e} ({delta[0]:+.2e})")


def get_available_correctors(field_components: Set[str], accel: str, ip: int,
                             optics_dfs: Sequence[DataFrame]) -> Sequence[IRCorrector]:
    """ Gets the available correctors by checking for their presence in the optics.
    If the corrector is not found in this ip, the ip is skipped.
    If only one corrector (left or right) is present a warning is issued.
    If one corrector is present in only one optics (and not in the other)
    an Environment Error is raised. """
    correctors = []
    for field_component in field_components:
        corrector_not_found = []
        corrector_names = []
        for side in SIDES:
            corrector = IRCorrector(field_component, accel, ip, side)
            corr_in_optics = [_corrector_in_optics(corrector.name, df) for df in optics_dfs]
            if all(corr_in_optics):
                correctors.append(corrector)
                corrector_names.append(corrector.name)
            elif any(corr_in_optics):
                # Raise informative Error
                idx_true = corr_in_optics.index(True)
                idx_false = (idx_true + 1) % 2
                raise EnvironmentError(f'Corrector {corrector.name} was found'
                                       f'in the {idx2str(idx_true + 1)} optics'
                                       f'but not in the {idx2str(idx_false + 1)}'
                                       f'optics.')
            else:
                # Ignore Corrector
                corrector_not_found.append(corrector.name)

        if len(corrector_not_found) == 1:
            LOG.warning(f'Corrector {corrector_not_found[0]} could not be found in '
                        f'optics, yet {corrector_names[0]} was present. '
                        f'Correction will be performed with available corrector(s) only.')
        elif len(corrector_not_found):
            LOG.info(f'Correctors {list2str(corrector_not_found)} were not found in'
                     f' optics. Skipping IP{ip}.')

    return list(sorted(correctors))  # do not have to be sorted, but looks nicer


def init_corrector_and_optics_values(correctors: Sequence[IRCorrector], optics_dfs: Sequence[DataFrame], update_optics: bool, ignore_settings: bool):
    """ If wished save original corrector values from optics (for later restoration)
    and sync corrector values in list and optics. """
    saved_values = {}

    for corrector in correctors:
        values = [df.loc[corrector.name, corrector.strength_component] for df in optics_dfs]

        if not update_optics:
            saved_values[corrector] = values

        if ignore_settings:
            # set corrector value in optics to zero
            for df in optics_dfs:
                df.loc[corrector.name, corrector.strength_component] = 0
        else:
            if any(np.diff(values)):
                raise ValueError(f"Initial value for corrector {corrector.name} differs "
                                 f"between optics.")
            # use optics value as initial value
            corrector.value = values[0]
    return saved_values


def restore_optics_values(saved_values: dict, optics_dfs: Sequence[DataFrame]):
    """ Restore saved initial corrector values (if any) to optics. """
    for corrector, values in saved_values.items():
        for df, val in zip(optics_dfs, values):
            df.loc[corrector.name, corrector.strength_component] = val


def get_elements_integral(rdt, ip, optics_df, errors_df, feeddown):
    """ Calculate the RDT integral for all elements of the IP. """
    integral = 0
    lm, jk = rdt.l + rdt.m, rdt.j + rdt.k
    # Integral on side ---
    for side in SIDES:
        LOG.debug(f" - Integral on side {side}.")
        side_sign = get_integral_sign(rdt.order, side)

        # get IP elements, errors and twiss have same elements because of check_dfs
        elements = optics_df.index[optics_df.index.str.match(fr".*{side}{ip:d}(\.B[12])?")]

        betax = optics_df.loc[elements, f"{BETA}{X}"]
        betay = optics_df.loc[elements, f"{BETA}{Y}"]
        if rdt.swap_beta_exp:
            # in case of beta-symmetry, this corrects for the same RDT in the opposite beam.
            betax = betax**(lm/2.)
            betay = betay**(jk/2.)
        else:
            betax = betax**(jk/2.)
            betay = betay**(lm/2.)

        dx = optics_df.loc[elements, X] + errors_df.loc[elements, f"{DELTA}{X}"]
        dy = optics_df.loc[elements, Y] + errors_df.loc[elements, f"{DELTA}{Y}"]
        dx_idy = dx + 1j*dy

        k_sum = Series(0j, index=elements)  # Complex sum of strengths (from K_n + iJ_n) and feed-down to them

        for q in range(feeddown+1):
            n_mad = rdt.order+q-1
            kl_opt = optics_df.loc[elements, f"K{n_mad:d}L"]
            kl_err = errors_df.loc[elements, f"K{n_mad:d}L"]
            iksl_opt = 1j*optics_df.loc[elements, f"K{n_mad:d}SL"]
            iksl_err = 1j*errors_df.loc[elements, f"K{n_mad:d}SL"]

            k_sum += ((kl_opt + kl_err + iksl_opt + iksl_err) *
                      (dx_idy**q) / np.math.factorial(q))

        integral += -sum(np.real(i_pow(lm) * k_sum.to_numpy()) * (side_sign * betax * betay).to_numpy())
    LOG.debug(f" -> Sum value: {integral}")
    return integral


def get_corrector_coefficient(rdt: RDT, corrector: IRCorrector, optics_df: DataFrame, errors_df: DataFrame):
    """ Calculate B-Matrix Element for Corrector. """
    LOG.debug(f" - Corrector {corrector.name}.")
    lm, jk = rdt.l + rdt.m, rdt.j + rdt.k

    sign_i = np.real(i_pow(lm + (lm % 2)))  # i_pow is always real
    sign_corrector = sign_i * get_integral_sign(rdt.order, corrector.side)

    betax = optics_df.loc[corrector.name, f"{BETA}{X}"]
    betay = optics_df.loc[corrector.name, f"{BETA}{Y}"]
    if rdt.swap_beta_exp:
        # in case of beta-symmetry, this corrects for the same RDT in the opposite beam.
        betax = betax**(lm/2.)
        betay = betay**(jk/2.)
    else:
        betax = betax**(jk/2.)
        betay = betay**(lm/2.)

    z = 1
    p = corrector.order - rdt.order
    if p:
        # Corrector contributes via feed-down
        dx = optics_df.loc[corrector.name, X] + errors_df.loc[corrector.name, f"{DELTA}{X}"]
        dy = optics_df.loc[corrector.name, Y] + errors_df.loc[corrector.name, f"{DELTA}{Y}"]
        dx_idy = dx + 1j*dy
        z_cmplx = (dx_idy**p) / np.math.factorial(p)
        if (corrector.skew and is_odd(lm)) or (not corrector.skew and is_even(lm)):
            z = np.real(z_cmplx)
        else:
            z = np.imag(z_cmplx)
            if not corrector.skew:
               z = -z
        if abs(z) < 1e-15:
            LOG.warning(f"Z-coefficient for {corrector.name} in {rdt.name} is very small.")

    return sign_corrector * z * betax * betay


def get_integral_sign(n: int, side: str) -> int:
    """ Sign of the integral and corrector for this side.

    This is the exp(iπnθ(s_w−s_IP)) part of Eq. (40) in
    [#DillyNonlinearIRCorrections2022]_,
    """
    if side == "R":
        # return (-1)**n
        return -1 if n % 2 else 1
    return 1


def _corrector_in_optics(name: str, df: DataFrame) -> bool:
    """ Checks if corrector is present in optics and has the KEYWORD MULTIPOLE. """
    if name not in df.index:
        LOG.debug(f'{name} not found in optics.')
        return False

    if KEYWORD in df.columns:
        if df.loc[name, KEYWORD].upper() != MULTIPOLE:
            LOG.warning(f"{name} found in optics, yet the Keyword was {df.loc[name, KEYWORD]}"
                        f" (should be '{MULTIPOLE}').")
            return False
    else:
        LOG.warning(f"'{KEYWORD}' column not found in optics."
                    f" Assumes you have filtered {MULTIPOLE}s beforehand!")

    return True


# Solve ---

def solve_equation_system(correctors: Sequence[IRCorrector], lhs: np.array, rhs: np.array, solver: str):
    """ Solves the system with the given solver.

    The result is transferred to the corrector values internally. """
    if len(rhs) > len(correctors) and solver not in APPROXIMATE_SOLVERS:
        raise ValueError("Overdetermined equation systems can only be solved "
                         "by one of the approximate solvers"
                         f" '{list2str(APPROXIMATE_SOLVERS)}'. "
                         f"Instead '{solver}' was chosen.")

    LOG.debug(f"Solving Equation system via {solver}.")
    # lhs x corrector = rhs -> correctors = lhs\rhs
    SOLVER_MAP[solver](correctors, lhs, rhs)


def _assign_corrector_values(correctors: Sequence[IRCorrector], values: Sequence):
    """ Assigns calculated values to the correctors. """
    for corr, val in zip(correctors, values):
        if len(val) > 1:
            raise ValueError(f"Multiple Values for corrector {str(corr)} found."
                             f" There should be only one.")
        LOG.debug(f"Updating Corrector: {str(corr)} {val[0]:+.2e}.")
        corr.value += val[0]
        LOG.info(str(corr))


# Update Optics ----------------------------------------------------------------

def update_optics(correctors: Sequence[IRCorrector], optics_dfs: Sequence[DataFrame]):
    """ Updates the corrector strength values in the current optics. """
    for optics in optics_dfs:
        for corrector in correctors:
            optics.loc[corrector.name, corrector.strength_component] = corrector.value


# Order Checks -----------------------------------------------------------------

def _check_corrector_order(rdt_maps: Sequence[dict], update_optics: bool, feed_down: int):
    """ Perform checks on corrector orders compared to RDT orders and feed-down. """
    for rdt_map in rdt_maps:
        for rdt, correctors in rdt_map.items():
            _check_corrector_order_not_lower(rdt, correctors)
            _check_update_optics(rdt, correctors, rdt_map, update_optics, feed_down)


def _check_update_optics(rdt: RDT, correctors: list, rdt_map: dict, update_optics: bool, feed_down: int):
    """ Check if corrector values are set before they would be needed for feed-down. """
    if not update_optics:
        return

    for corrector in correctors:
        corrector_order = int(corrector[1])
        for rdt_comp in rdt_map.keys():
            if rdt_comp.order == rdt.order:
                break
            if rdt_comp.order <= corrector_order <= rdt_comp.order + feed_down:
                raise ValueError(
                    "Updating the optics is in this configuration not possible,"
                    f" as corrector {corrector} influences {rdt_comp.name}"
                    f" with the given feeddown of {feed_down}. Yet the value of"
                    f" the corrector is defined by {rdt.name}.")


def _check_corrector_order_not_lower(rdt, correctors):
    """ Check if only higher and equal order correctors are defined to correct
    a given rdt."""
    wrong_correctors = [c for c in correctors if int(c[1]) < rdt.order]
    if len(wrong_correctors):
        raise ValueError(
            "Correctors can not correct RDTs of higher order."
            f" Yet for {rdt.name} the corrector(s)"
            f" '{list2str(wrong_correctors)}' was (were) given."
        )


# IO Handling ------------------------------------------------------------------
# Input ---

def _check_opt(opt: DotDict) -> DotDict:
    # check for unkown input
    parser = get_parser()
    known_opts = [a.dest for a in parser._actions if not isinstance(a, argparse._HelpAction)]  # best way I could figure out
    unknown_opts = [k for k in opt.keys() if k not in known_opts]
    if len(unknown_opts):
        raise AttributeError(f"Unknown arguments found: '{list2str(unknown_opts)}'.\n"
                             f"Allowed input parameters are: '{list2str(known_opts)}'")

    # Set defaults
    for name, default in DEFAULTS.items():
        if opt.get(name) is None:
            LOG.debug(f"Setting input '{name}' to default value '{default}'.")
            opt[name] = default

    # check accel
    opt.accel = opt.accel.lower()  # let's not care about case
    if opt.accel not in DEFAULT_RDTS.keys():
        raise ValueError(f"Parameter 'accel' needs to be one of '{list2str(list(DEFAULT_RDTS.keys()))}' "
                         f"but was '{opt.accel}' instead.")

    # Set rdts:
    if opt.get('rdts') is None:
        opt.rdts = DEFAULT_RDTS[opt.accel]

    # Check required and rdts:
    for name in ('optics', 'errors', 'beams', 'rdts'):
        inputs = opt.get(name)
        if inputs is None or isinstance(inputs, str) or not isinstance(inputs, (Iterable, Sized)):
            raise ValueError(f"Parameter '{name}' is required and needs to be "
                             "iterable, even if only of length 1. "
                             f"Instead was '{inputs}'.")

    # Copy DataFrames as they might be modified
    opt.optics = [o.copy() for o in opt.optics]
    opt.errors = [e.copy() for e in opt.errors]

    if opt.feeddown < 0 or not (opt.feeddown == int(opt.feeddown)):
        raise ValueError("'feeddown' needs to be a positive integer.")

    if opt.iterations < 1:
        raise ValueError("At least one iteration (see: 'iterations') needs to "
                         "be done for correction.")
    return opt


def get_tfs(paths: Sequence) -> Sequence[tfs.TfsDataFrame]:
    if isinstance(paths[0], str) or isinstance(paths[0], Path):
        return tuple(tfs.read_tfs(path, index="NAME") for path in paths)
    return paths


def _check_dfs(optics_dfs, errors_dfs, beams, orders, ignore_missing_columns):
    if len(optics_dfs) > 2 or len(errors_dfs) > 2:
        raise NotImplementedError("A maximum of two optics can be corrected "
                                  "at the same time, for now.")

    if len(optics_dfs) != len(errors_dfs):
        raise ValueError(f"The number of given optics ({len(optics_dfs):d}) "
                         "does not equal the number of given error files "
                         f"({len(errors_dfs):d}). Hint: it should.")

    if len(optics_dfs) != len(beams):
        raise ValueError(f"The number of given optics ({len(optics_dfs):d}) "
                         "does not equal the number of given beams "
                         f"({len(beams):d}). Please specify a beam for each "
                         "optics.")

    for idx_file, (optics, errors) in enumerate(zip(optics_dfs, errors_dfs)):
        not_found_errors = errors.index.difference(optics.index)
        if len(not_found_errors):
            raise IOError("The following elements were found in the "
                          f"{idx2str(idx_file)} given errors file but not in"
                          f"the optics: {list2str(not_found_errors.to_list())}")

        not_found_optics = optics.index.difference(errors.index)
        if len(not_found_optics):
            LOG.debug("The following elements were found in the "
                      f"{idx2str(idx_file)} given optics file but not in "
                      f"the errors: {list2str(not_found_optics.to_list())}."
                      f"They are assumed to be zero.")
            for indx in not_found_optics:
                errors.loc[indx, :] = 0

        needed_values = [f"K{order-1:d}{orientation}L"  # -1 for madx-order
                         for order in orders
                         for orientation in ("S", "")]

        for df, file_type in ((optics, "optics"), (errors, "error")):
            not_found_strengths = [s for s in needed_values if s not in df.columns]
            if len(not_found_strengths):
                text = ("Some strength values were not found in the "
                        f"{idx2str(idx_file)} given {file_type} file: "
                        f"{list2str(not_found_strengths)}.")

                if not ignore_missing_columns:
                    raise IOError(text)
                LOG.warning(text + " They are assumed to be zero.")
                for kl in not_found_strengths:
                    df[kl] = 0


def switch_signs_for_beam4(optics_dfs: Iterable[tfs.TfsDataFrame],
                           error_dfs: Iterable[tfs.TfsDataFrame], beams: Iterable[int]):
    """ Switch the signs for Beam 4 optics.
     This is due to the switch in direction for this beam and
     (anti-) symmetry after a rotation of 180deg around the y-axis of magnets,
     combined with the fact that the KL values in MAD-X twiss do not change sign,
     but in the errors they do (otherwise it would compensate).
     Magnet orders that show anti-symmetry are: a1 (K0SL), b2 (K1L), a3 (K2SL), b4 (K3L) etc.
     Also the sign for (delta) X is switched back to have the same orientation as beam2."""
    for idx, (optics_df, error_df, beam) in enumerate(zip(optics_dfs, error_dfs, beams)):
        if beam == 4:
            LOG.debug(f"Beam 4 input found. Switching signs for X and K(S)L values when needed.")
            optics_df[X] = -optics_df[X]
            error_df[f"{DELTA}{X}"] = -error_df[f"{DELTA}{X}"]

            max_order = error_df.columns.str.extract(r"^K(\d+)S?L$", expand=False).dropna().astype(int).max()
            for order in range(max_order+1):
                name = f"K{order:d}{SKEW_NAME_MAP[is_even(order)]}L"
                if name in error_df.columns:
                    error_df[name] = -error_df[name]


# Output --------

def get_and_write_output(out_path: str, correctors: Sequence) -> Tuple[str, tfs.TfsDataFrame]:
    correction_text = build_correction_str(correctors)
    correction_df = build_correction_df(correctors)

    if out_path is not None:
        out_path = Path(out_path)
        write_command(out_path, correction_text)
        write_tfs(out_path, correction_df)

    return correction_text, correction_df


# Build ---

def build_correction_df(correctors: Sequence) -> tfs.TfsDataFrame:
    attributes = vars(correctors[0])
    return tfs.TfsDataFrame(
        data=[[getattr(cor, attr) for attr in attributes] for cor in correctors],
        columns=attributes,
    )


def build_correction_str(correctors: Sequence) -> str:
    return _build_correction_str(corr for corr in correctors)


def build_correction_str_from_df(correctors_df: DataFrame) -> str:
    return _build_correction_str(row[1] for row in correctors_df.iterrows())


def _build_correction_str(iterator: Iterable) -> str:
    """ Creates madx commands (assignments) to run for correction"""
    last_type = ''
    text = ''
    for corr in iterator:
        if not _types_almost_equal(corr.type, last_type):
            text += f"\n!! {_nice_type(corr.type)} ({corr.field_component}) corrector\n"
            last_type = corr.type
        text += f"{corr.circuit} := {corr.value: 6E} / l.{corr.type} ;\n"
    return text.lstrip("\n")


def _types_almost_equal(type_a: str, type_b: str) -> bool:
    """ Groups correctors with and without ending F. """
    if type_a == type_b:
        return True
    if type_a.endswith('F'):
        return type_a[:-1] == type_b
    if type_b.endswith('F'):
        return type_b[:-1] == type_a


def _nice_type(mtype: str) -> str:
    if mtype.endswith('F'):
        return f'{mtype[:-1]}[F]'
    return mtype


# Write ---

def write_command(out_path: Path, correction_text: str):
    out_path_cmd = out_path.with_suffix(EXT_MADX)
    with open(out_path_cmd, "w") as out:
        out.write(correction_text)


def write_tfs(out_path: Path, correction_df: tfs.TfsDataFrame):
    out_path_df = out_path.with_suffix(EXT_TFS)
    tfs.write(out_path_df, correction_df)


# Helper -----------------------------------------------------------------------

def list2str(list_: list) -> str:
    return str(list_).strip('[]')


def idx2str(idx: int) -> str:
    return {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}[idx+1]


def order2field_component(order: int, skew: bool) -> str:
    return f"{SKEW_FIELD_MAP[skew]:s}{order:d}"


def field_component2order(field_component) -> Tuple[int, bool]:
    return int(field_component[1]), FIELD_SKEW_MAP[field_component[0]]


def is_odd(n):
    return bool(n % 2)


def is_even(n):
    return not is_odd(n)


def i_pow(n):
    return 1j**(n % 4)   # more exact with modulo


# Script Mode ------------------------------------------------------------------


def log_setup():
    """ Set up a basic logger. """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)7s | %(message)s | %(name)s"
    )


if __name__ == '__main__':
    log_setup()
    main()
