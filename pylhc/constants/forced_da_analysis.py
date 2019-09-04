RESULTS_DIR = "forced_da_analysis"

ROLLING_AVERAGE_WINDOW = 7
FILL_TIME_AROUND_KICKS_MIN = 10
TIME_BEFORE_KICK_S = (-30, -5)
TIME_AFTER_KICK_S = (5, 30)

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

def get_intensity_plotfile(plane, ftype):
    return f"intensity_{plane.lower()}{ftype}"


def get_emittance_plotfile(plane, ftype):
    return f"emittance_{plane.lower()}{ftype}"


def get_losses_plotfile(plane, ftype):
    return f"losses_{plane.lower()}{ftype}"


def get_da_fit_plotfile(plane, ftype):
    return f"dafit_{plane.lower()}{ftype}"


PLOT_NAMEMAP = dict(
    intensity=get_intensity_plotfile,
    emittance=get_emittance_plotfile,
    losses=get_losses_plotfile,
    dafit=get_da_fit_plotfile,
)

# Timber Keys ------------------------------------------------------------------

INTENSITY_KEY = 'LHC.BCTFR.A6R4.B{beam:d}:BEAM_INTENSITY'
BUNCH_EMITTANCE_KEY = 'LHC.BSRT.5R4.B{beam:d}:BUNCH_EMITTANCE_{plane:s}'
BWS_EMITTANCE_KEY = 'LHC.BWS.5R4.B{beam:d}{plane:s}.APP.{direction:s}:EMITTANCE_NORM'

# Headers ----------------------------------------------------------------------

HEADER_TIME_BEFORE = "Timespan before kick (for intensity averaging) [s]"
HEADER_TIME_AFTER = "Timespan after kick (for intensity averaging) [s]"
HEADER_EMITTANCE_AVERAGE = "Rolling window length for BSRT-Emittance averaging"
HEADER_DA = "Forced DA Fit"
HEADER_DA_ERROR = "Forced DA Fit Error"


def header_da(plane):
    return f"{HEADER_DA} {plane.upper()}"


def header_da_error(plane):
    return f"{HEADER_DA_ERROR} {plane.upper()}"


# Columns ----------------------------------------------------------------------

INTENSITY = "INTENSITY"
INTENSITY_BEFORE = "I_BEFORE"
INTENSITY_AFTER = "I_AFTER"
INTENSITY_LOSSES = "I_LOSSES"
SIGMA = "SIGMA"

# Column Modifiers ---
CLEAN = "CLEAN"
MEAN = "MEAN"
ERR = "ERR"
RELATIVE = "REL"


def err_col(column):
    return f"{ERR}{column}"


def mean_col(column):
    return f"{MEAN}{column}"


def rel_col(column):
    return f"{column}{RELATIVE}"

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
