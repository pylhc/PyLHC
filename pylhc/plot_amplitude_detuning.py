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
from generic_parser.entry_datatypes import DictAsString
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

# noinspection PyUnresolvedReferences
from tfs.tools import significant_digits

from pylhc import omc3_context
from pylhc.omc3.omc3.tune_analysis import constants as const, kick_file_modifiers as kick_mod
from pylhc.omc3.omc3.utils import logging_tools, plot_style as pstyle
from pylhc.plotshop import colors as pcolors, annotations as pannot

LOG = logging_tools.get_logger(__name__)

PLANES = const.get_planes()
COL_MAV = const.get_mav_col
COL_IN_MAV = const.get_used_in_mav_col
COL_BBQ = const.get_bbq_col
COL_TIME = const.get_time_col
HEADER_MAV_WINDOW = const.get_mav_window_header

UNIT_TO_UM = dict(km=1e9, m=1e6, mm=1e3, um=1, nm=1e-3, pm=1e-6, fm=1e-9,)


MANUAL_STYLE_DEFAULT= {
    u'figure.figsize': [9.5, 4],
    u"lines.marker": u"o",
    u"lines.linestyle": u"",
}


def get_params():
    return EntryPointParameters(
        kicks=dict(
            nargs="+",
            help="Kick files as data frames or tfs files.",
            required=True,
        ),
        labels=dict(
            help="Labels for the data. Needs to be same length as kicks.",
            nargs='+',
            required=True,
            type=str,
        ),
        plane=dict(
            help="Plane of the kicks.",
            required=True,
            choices=PLANES,
            type=str,
        ),
        correct_acd=dict(
            help="Correct for AC-Dipole kicks.",
            action="store_true",
        ),
        output=dict(
            help="Save the amplitude detuning plot here.",
            type=str,
        ),
        show=dict(
            help="Show the amplitude detuning plot.",
            action="store_true",
        ),
        ymin=dict(
            help="Minimum tune in units of tune scale (y-axis) in amplitude detuning plot.",
            type=float,
        ),
        ymax=dict(
            help="Maximum tune in units of tune scale (y-axis) in amplitude detuning plot.",
            type=float,
        ),
        xmin=dict(
            help="Minimum action in um (x-axis) in amplitude detuning plot.",
            type=float,
        ),
        xmax=dict(
            help="Maximum action in um (x-axis) in amplitude detuning plot.",
            type=float,
        ),
        action_unit=dict(
            help="Unit the action is given in.",
            default="m",
            choices=list(UNIT_TO_UM.keys()),
            type=str,
        ),
        manual_style=dict(
            help="Additional plotting style.",
            type=DictAsString,
            default={}
        ),
        tune_scale=dict(
            help="Plotting exponent of the tune.",
            default=-3,
            type=int,
        )
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Plotting Amplitude Detuning Results.")
    # todo: save opt
    _check_opt(opt)

    kick_plane = opt.plane
    figs = {}

    _set_plotstyle(opt.manual_style)
    limits = opt.get_subdict(['xmin', 'xmax', 'ymin', 'ymax'])

    for tune_plane in PLANES:
        for corr in [False, True]:
            corr_label = "_corrected" if corr else ""
            acd_corr = 1
            if opt.correct_acd and (kick_plane == tune_plane):
                acd_corr = 0.5

            fig = plt.figure()
            ax = fig.add_subplot(111)

            for idx, (kick, label) in enumerate(zip(opt.kicks, opt.labels)):
                kick_df = kick_mod.read_timed_dataframe(kick) if isinstance(kick, str) else kick

                data = kick_mod.get_ampdet_data(kick_df, kick_plane, tune_plane, corr)
                odr_fit = kick_mod.get_linear_odr_data(kick_df, kick_plane, tune_plane, corr)
                data, odr_fit, scale_to_m = _scale_and_correct_data_to_um(data, odr_fit,
                                                                          opt.action_unit, 10**opt.tune_scale, acd_corr)

                _plot_detuning(ax, data=data, label=label, limits=limits,
                               odr_fit=odr_fit,
                               color=pcolors.get_mpl_color(idx),
                               scale_to_m=scale_to_m, tune_scale=10**opt.tune_scale)

            ax_labels = const.get_paired_lables(kick_plane, tune_plane, opt.tune_scale)
            id_str = f"Q{tune_plane.upper():s}J{kick_plane.upper():s}{corr_label:s}"
            pannot.set_name(id_str, fig)
            _format_axes(ax, labels=ax_labels, limits=limits)

            if opt.show:
                plt.show()

            if opt.output:
                output = os.path.splitext(opt.output)
                output = "{:s}_{:s}{:s}".format(output[0], id_str, output[1])
                fig.savefig(output)

            plt.close(fig)
            figs[id_str] = fig

    return figs


# Other Functions --------------------------------------------------------------

def _check_opt(opt):
    if len(opt.labels) != len(opt.kicks):
        raise ValueError("'kicks' and 'labels' need to be of same size!")


def _set_plotstyle(manual_style):
    mstyle = MANUAL_STYLE_DEFAULT
    mstyle.update(manual_style)
    pstyle.set_style("standard", mstyle)


def _get_linear_odr_label(slope, std, scale_to_m, tune_scale):
    scale = scale_to_m  * tune_scale * 1e-3
    s_slope, s_std = significant_digits(slope*scale, std*scale)
    return f'({s_slope} $\pm$ {s_std}) $\cdot$ 10$^3$ m$^{{-1}}$'


def plot_linear_odr(ax, odr_fit, xmax, scale_to_m, tune_scale, color=None):
    """ Adds a linear odr fit to axes.
    """
    offset, slope, slope_std = odr_fit.beta + [odr_fit.sd_beta[1]]
    x = [0, xmax]
    y = [0, xmax * slope]
    y_low = [0, x[1] * (slope - slope_std)]
    y_upp = [0, x[1] * (slope + slope_std)]
    label = _get_linear_odr_label(slope, slope_std, scale_to_m, tune_scale)
    color = 'k' if color is None else color

    ax.fill_between(x, y_low, y_upp, facecolor=mcolors.to_rgba(color, .3))
    ax.plot(x, y, marker="", linestyle='--', color=color, label=label)

    return offset


def _plot_detuning(ax, data, label="", scale_to_m=1., tune_scale=1., color=None,
                   limits=None, odr_fit=None):
    """ Plot the detuning and the ODR into axes. """
    xmax = _get_default(limits, 'xmax', max(data['x']+data['xerr']))
    offset = 0
    if odr_fit:
        offset = plot_linear_odr(ax, odr_fit, xmax=xmax, scale_to_m=scale_to_m, tune_scale=tune_scale, color=color)

    data['y'] -= offset
    ax.errorbar(**data, label=label, color=color)


def _format_axes(ax, limits, labels):
    # labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    # limits
    ax.set_xlim(left=limits['xmin'], right=limits['xmax'])
    ax.set_ylim(bottom=limits['ymin'], top=limits['ymax'])

    pannot.make_top_legend(ax, ncol=2)

    ax.figure.tight_layout()
    ax.figure.tight_layout()  # needs two calls for some reason to look great


def _get_default(ddict, key, default):
    """ Returns 'default' if either the dict itself or the entry is None. """
    if ddict is None or key not in ddict or ddict[key] is None:
        return default
    return ddict[key]


def _scale_and_correct_data_to_um(data, odr_fit, unit, tune_scale, acd_corr):
    # scale units
    scale = UNIT_TO_UM[unit]
    data['x'] *= scale
    data['xerr'] *= scale
    odr_fit.beta[1] /= scale
    odr_fit.sd_beta[1] /= scale

    # correct for ac-diple and tune units
    acd_corr /= tune_scale
    data['y'] *= acd_corr
    data['yerr'] *= acd_corr

    odr_fit.beta[0] *= acd_corr
    odr_fit.beta[1] *= acd_corr
    odr_fit.sd_beta[1] *= acd_corr
    return data, odr_fit, 1e6


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()

