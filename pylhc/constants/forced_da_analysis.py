from tfs.tools import DotDict

RESULTS_DIR = "forced_da_analysis"

ROLLING_AVERAGE_WINDOW = 7
FILL_TIME_AROUND_KICKS_MIN = 10
TIME_BEFORE_KICK_S = (-30, -5)
TIME_AFTER_KICK_S = (5, 30)

INITIAL_DA_FIT = 4  # initial DA for fitting in values of nominal emittance

BWS_DIRECTIONS = ("IN", "OUT")

# Kick File Definitions --------------------------------------------------------

KICKFILE = "kick"
TFS_SUFFIX = ".tfs"
TIME = "TIME"


def get_kick_outfile(plane):
    return f'{KICKFILE}_fda_{plane.lower()}{TFS_SUFFIX}'


def get_emittance_outfile(plane):
    return f'emittance_{plane.lower()}{TFS_SUFFIX}'


def get_emittance_bws_outfile(plane):
    return f'emittance_bws_{plane.lower()}{TFS_SUFFIX}'


# Plotting ---------------------------------------------------------------------

PLOT_FILETYPES = (".pdf", ".png")


def get_plot_filename(ptype, plane, ftype):
    return f"{ptype}_{plane.lower()}{ftype}"


# Timber Keys ------------------------------------------------------------------

INTENSITY_KEY = 'LHC.BCTFR.A6R4.B{beam:d}:BEAM_INTENSITY'

BUNCH_EMITTANCE_KEY = 'LHC.BSRT.5R4.B{beam:d}:AVERAGE_EMITTANCE_{plane:s}'
BUNCH_EMITTANCE_TO_METER = 1e-6  # Emittance is normalized an in um

BWS_EMITTANCE_KEY = 'LHC.BWS.5R4.B{beam:d}{plane:s}.APP.{direction:s}:EMITTANCE_NORM'
BWS_EMITTANCE_TO_METER = 1e-6  # Emittance is normalized an in um

# Headers ----------------------------------------------------------------------

HEADER_TIME_BEFORE = "Timespan before kick (for intensity averaging) [s]"
HEADER_TIME_AFTER = "Timespan after kick (for intensity averaging) [s]"
HEADER_EMITTANCE_AVERAGE = "Rolling window length for BSRT-Emittance averaging"
HEADER_DA = "Forced DA Fit {plane:} [{unit:}]"
HEADER_DA_ERROR = "Forced DA Fit Error {plane:} [{unit:}]"
HEADER_ENERGY = "Beam Energy [GeV]"
HEADER_NOMINAL_EMITTANCE = "Nominal Emittance {plane:}[m]"


def header_da(plane, unit="m"):
    return HEADER_DA.format(plane=plane.upper(), unit=unit)


def header_da_error(plane, unit="m"):
    return HEADER_DA_ERROR.format(plane=plane.upper(), unit=unit)


def header_nominal_emittance(plane):
    return HEADER_EMITTANCE_AVERAGE.format(plane.upper())


def header_norm_nominal_emittance(plane):
    return f"Normalized {HEADER_EMITTANCE_AVERAGE.format(plane.upper())}"

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


def err_col(column):
    return f"{ERR}{column}"


def mean_col(column):
    return f"{MEAN}{column}"


def rel_col(column):
    return f"{column}{RELATIVE}"


def sigma_col(column):
    return f"{SIGMA}{column}"


# Planed columns ---
EMITTANCE = "EMITTANCE"
NORM_EMITTANCE = "NORMEMITTANCE"


def column_action(plane):
    return f"2J{plane.upper()}RES"


def column_emittance(plane):
    return f"{EMITTANCE}{plane.upper()}"


def column_norm_emittance(plane):
    return f"{NORM_EMITTANCE}{plane.upper()}"


def column_sigma(plane):
    return f"{SIGMA}{plane.upper()}"
