"""
Constants: Forced DA Analysis
-----------------------------

Specific constants relating to the forced DA analysis to be used in ``PyLHC``, to help with
consistency.
"""
from pylhc.constants.general import PLANE_TO_HV, TFS_SUFFIX

RESULTS_DIR = "forced_da_analysis"

ROLLING_AVERAGE_WINDOW = 100
OUTLIER_LIMIT = 0.5 * 1e-6
TIME_AROUND_KICKS_MIN = 10
TIME_BEFORE_KICK_S = [30, 5]
TIME_AFTER_KICK_S = [5, 30]
YPAD = 0.05  # additional padding of the y axis for DA plots

INITIAL_DA_FIT = 12  # initial DA for fitting in values of nominal emittance
MAX_CURVEFIT_FEV = 10000  # Max number of curve_fit iterations


BWS_DIRECTIONS = ("IN", "OUT")


# Kick File Definitions --------------------------------------------------------

KICKFILE = "kick"


def outfile_kick(plane) -> str:
    return f"{KICKFILE}_fda_{plane.lower()}{TFS_SUFFIX}"


def outfile_emittance(plane) -> str:
    return f"emittance_{plane.lower()}{TFS_SUFFIX}"


def outfile_emittance_bws(plane) -> str:
    return f"emittance_bws_{plane.lower()}{TFS_SUFFIX}"


OUTFILE_INTENSITY = f"intensity{TFS_SUFFIX}"

# Plotting ---------------------------------------------------------------------

PLOT_FILETYPES = (".pdf", ".png")


def outfile_plot(ptype, plane, ftype) -> str:
    return f"{ptype}_{plane.lower()}{ftype}"


# Timber Keys ------------------------------------------------------------------

INTENSITY_KEY = "LHC.BCTFR.A6R4.B{beam:d}:BEAM_INTENSITY"

BSRT_EMITTANCE_SIGMA_FIT_KEY = "LHC.BSRT.5{side:s}4.B{beam:d}:FIT_SIGMA_{plane:s}"
BSRT_EMITTANCE_AVERAGE_KEY = "LHC.BSRT.5{side:s}4.B{beam:d}:AVERAGE_EMITTANCE_{plane:s}"
BSRT_EMITTANCE_TO_METER = 1e-6  # Emittance is normalized and in um

BWS_EMITTANCE_KEY = "LHC.BWS.5{side:s}4.B{beam:d}{plane:s}.APP.{direction:s}:EMITTANCE_NORM"
BWS_EMITTANCE_TO_METER = 1e-6  # Emittance is normalized and in um


LR_MAP = {1: "R", 2: "L"}


def bsrt_emittance_key(beam, plane, type_):
    key = {"fit_sigma": BSRT_EMITTANCE_SIGMA_FIT_KEY, "average": BSRT_EMITTANCE_AVERAGE_KEY}[type_]
    return key.format(side=LR_MAP[beam], beam=beam, plane=PLANE_TO_HV[plane])


def bws_emittance_key(beam, plane, direction) -> str:
    return BWS_EMITTANCE_KEY.format(
        side=LR_MAP[beam], beam=beam, plane=PLANE_TO_HV[plane], direction=direction
    )


# Headers ----------------------------------------------------------------------
HEADER_TIME_BEFORE = "Timespan_before_kick[s]"
HEADER_TIME_AFTER = "Timespan_after_kick[s]"
HEADER_BSRT_ROLLING_WINDOW = "Emittance_rolling_window_length"
HEADER_BSRT_OUTLIER_LIMIT = "Emittance_outlier_limit"
HEADER_DA = "Forced_DA_J_Fit_{plane:}[{unit:}]"
HEADER_DA_ERROR = "Forced_DA_J_Fit_Error_{plane:}[{unit:}]"
HEADER_ENERGY = "Beam_Energy[GeV]"
HEADER_NOMINAL_EMITTANCE = "Nominal_Emittance_{plane:}[m]"


def header_da(plane, unit="m") -> str:
    return HEADER_DA.format(plane=plane.upper(), unit=unit)


def header_da_error(plane, unit="m") -> str:
    return HEADER_DA_ERROR.format(plane=plane.upper(), unit=unit)


def header_nominal_emittance(plane) -> str:
    return HEADER_NOMINAL_EMITTANCE.format(plane=plane.upper())


def header_norm_nominal_emittance(plane) -> str:
    return f"Normalized_{HEADER_NOMINAL_EMITTANCE.format(plane=plane.upper())}"


# Columns ----------------------------------------------------------------------

INTENSITY = "INTENSITY"
INTENSITY_BEFORE = "I_BEFORE"
INTENSITY_AFTER = "I_AFTER"
INTENSITY_LOSSES = "I_LOSSES"

# Column Modifiers ---
CLEAN = "CLEAN"
MEAN = "MEAN"
ERR = "ERR"
RELATIVE = "REL"
SIGMA = "SIGMA"


def err_col(column) -> str:
    return f"{ERR}{column}"


def mean_col(column) -> str:
    return f"{MEAN}{column}"


def rel_col(column) -> str:
    return f"{column}{RELATIVE}"


def sigma_col(column) -> str:
    return f"{SIGMA}{column}"


# Planed columns ---
EMITTANCE = "EMITTANCE"
NORM_EMITTANCE = "NORMEMITTANCE"


def column_action(plane) -> str:
    return f"2J{plane.upper()}RES"


def column_emittance(plane) -> str:
    return f"{EMITTANCE}{plane.upper()}"


def column_norm_emittance(plane) -> str:
    return f"{NORM_EMITTANCE}{plane.upper()}"


def column_bws_norm_emittance(plane, direction) -> str:
    return f"{column_norm_emittance(plane)}_{direction}"
