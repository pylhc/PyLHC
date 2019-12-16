"""
Amplitude Detuning Results Plotting
------------------------------------

Provides the plotting function for amplitude detuning analysis


:module: plot_amplitude_detuning
:author: jdilly

"""
import os

import numpy as np
import tfs
from generic_parser import entrypoint, EntryPointParameters
from matplotlib import pyplot as plt

# noinspection PyUnresolvedReferences
from pylhc import omc3_context
from pylhc.omc3.omc3.tune_analysis import constants as const, kick_file_modifiers as kick_mod
from pylhc.omc3.omc3.utils import logging_tools, plot_style as pstyle

LOG = logging_tools.get_logger(__name__)

PLANES = const.get_planes()
COL_MAV = const.get_mav_col
COL_IN_MAV = const.get_used_in_mav_col
COL_BBQ = const.get_bbq_col
COL_TIME = const.get_time_col
HEADER_MAV_WINDOW = const.get_mav_window_header


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        flags="--kick",
        help="Kick file as data frame or tfs file.",
        name="kick",
        required=True,
    )
    params.add_parameter(
        flags="--plane",
        help="Plane of the kicks. 'X' or 'Y'.",
        name="plane",
        required=True,
        choices=PLANES,
        type=str,
    )
    params.add_parameter(
        flags="--out",
        help="Save the amplitude detuning plot here.",
        name="output",
        type=str,
    )
    params.add_parameter(
        flags="--label",
        help="Label for the data.",
        name="label",
        type=str,
    )
    params.add_parameter(
        flags="--show",
        help="Show the amplitude detuning plot.",
        name="show",
        action="store_true",
    )
    params.add_parameter(
        flags="--ymin",
        help="Minimum tune (y-axis) in amplitude detuning plot.",
        name="ymin",
        type=float,
    )
    params.add_parameter(
        flags="--ymax",
        help="Maximum tune (y-axis) in amplitude detuning plot.",
        name="ymax",
        type=float,
    )
    params.add_parameter(
        flags="--xmin",
        help="Minimum action (x-axis) in amplitude detuning plot.",
        name="xmin",
        type=float,
    )
    params.add_parameter(
        flags="--xmax",
        help="Maximum action (x-axis) in amplitude detuning plot.",
        name="xmax",
        type=float,
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Plotting Amplitude Detuning Results.")
    kick_df = tfs.read(opt.kick, index=COL_TIME())if isinstance(opt.kick, str) else opt.kick
    opt.pop("kick")

    show = opt.pop("show")
    out = opt.pop("output")
    kick_plane = opt.pop("plane")
    label = opt.pop("label")

    figs = {}

    for tune_plane in PLANES:
        for corr in [False, True]:
            corr_label = "_corrected" if corr else ""

            data = kick_mod.get_ampdet_data(kick_df, kick_plane, tune_plane, corr)
            odr_fit = kick_mod.get_linear_odr_data(kick_df, kick_plane, tune_plane, corr)

            labels = const.get_paired_lables(kick_plane, tune_plane)
            id_str = "J{:s}_Q{:s}{:s}".format(kick_plane.upper(), tune_plane.upper(), corr_label)

            fig = plot_detuning(**data,
                                odr_fit=odr_fit,
                                odr_plot=plot_linear_odr,
                                labels={"x": labels[0], "y": labels[1], "line": label},
                                **opt,
                                )

            if show:
                plt.show()

            if out:
                output = os.path.splitext(out)
                output = "{:s}_{:s}{:s}".format(output[0], id_str, output[1])
                fig.savefig(output)

            plt.close(fig)
            figs[id_str] = fig

    return figs


# Other Functions --------------------------------------------------------------


def plot_linear_odr(ax, odr_fit, lim):
    """ Adds a linear odr fit to axes.
    """
    x_fit = np.linspace(lim[0], lim[1], 2)
    line_fit = odr_fit.beta[1] * x_fit
    ax.plot(x_fit, line_fit, marker="", linestyle='--', color='k',
            label='${:.4f}\, \pm\, {:.4f}$'.format(odr_fit.beta[1], odr_fit.sd_beta[1]))


def plot_detuning(x, y, xerr, yerr, labels, xmin=None, xmax=None, ymin=None, ymax=None,
                  odr_fit=None, odr_plot=plot_linear_odr):
    """ Plot amplitude detuning.

    Args:
        x: Action data.
        y: Tune data.
        xerr: Action error.
        yerr: Tune error.
        xmin: Lower action range to plot.
        xmax: Upper action range to plot.
        ymin: Lower tune range to plot.
        ymax: Upper tune range to plot.
        odr_fit: results of the odr-fit (e.g. see do_linear_odr)
        odr_plot: function to plot odr_fit (e.g. see plot_linear_odr)
        labels: Dict of labels to use for the data ("line"), the x-axis ("x") and the y-axis ("y")

    Returns:
        Plotted Figure
    """
    pstyle.set_style("standard",
                 {u"lines.marker": u"o",
                  u"lines.linestyle": u"",
                  u'figure.figsize': [9.5, 4],
                  }
                 )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xmin = 0 if xmin is None else xmin
    xmax = max(x + xerr) * 1.05 if xmax is None else xmax

    offset = 0
    if odr_fit:
        odr_plot(ax, odr_fit, lim=[xmin, xmax])
        offset = odr_fit.beta[0]

    ax.errorbar(x, y - offset, xerr=xerr, yerr=yerr, label=labels.get("line", None))

    # labels
    default_labels = const.get_paired_lables("", "")
    ax.set_xlabel(labels.get("x", default_labels[0]))
    ax.set_ylabel(labels.get("y", default_labels[1]))

    # limits
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    # lagends
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2,)
    ax.ticklabel_format(style="sci", useMathText=True, scilimits=(-3, 3))
    fig.tight_layout()
    fig.tight_layout()  # needs two calls for some reason to look great
    return fig


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()

