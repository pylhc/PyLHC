"""
Amplitude Detuning Results Plotting
------------------------------------

Provides the plotting function for amplitude detuning analysis


:module: plot_amplitude_detuning
:author: jdilly

"""
import os
from functools import partial

import numpy as np
import tfs
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entry_datatypes import DictAsString
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

# noinspection PyUnresolvedReferences
from tfs.tools import significant_digits

from pylhc import omc3_context
from pylhc.omc3.omc3.tune_analysis import constants as const, kick_file_modifiers as kick_mod, detuning_tools
from pylhc.omc3.omc3.utils import logging_tools, plot_style as pstyle
from pylhc.plotshop import colors as pcolors, annotations as pannot
from pylhc.constants.general import UNIT_TO_M

LOG = logging_tools.get_logger(__name__)

PLANES = const.get_planes()
COL_MAV = const.get_mav_col
COL_IN_MAV = const.get_used_in_mav_col
COL_BBQ = const.get_bbq_col
COL_TIME = const.get_time_col
HEADER_MAV_WINDOW = const.get_mav_window_header


NFIT = 100  # Points for the fitting function

MANUAL_STYLE_DEFAULT = {
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
        detuning_order=dict(
            help="Order of the detuning as int. Basically just the order of the applied fit.",
            type=int,
            default=1,
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
            choices=list(UNIT_TO_M.keys()),
            type=str,
        ),
        action_plot_unit=dict(
            help="Unit the action should be plotted in.",
            default="um",
            choices=list(UNIT_TO_M.keys()),
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

                data = kick_mod.get_ampdet_data(kick_df, kick_plane, tune_plane, corrected=corr)
                odr_fit = kick_mod.get_odr_data(kick_df, kick_plane, tune_plane,
                                                order=opt.detuning_order, corrected=corr)
                data, odr_fit = _correct_and_scale(data, odr_fit,
                                                   opt.action_unit, opt.action_plot_unit,
                                                   10**opt.tune_scale, acd_corr)

                _plot_detuning(ax, data=data, label=label, limits=limits,
                               odr_fit=odr_fit,
                               color=pcolors.get_mpl_color(idx),
                               action_unit=opt.action_plot_unit, tune_scale=10**opt.tune_scale)

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


# Plotting --------------------------------------------------------------


def _get_scaled_odr_label(odr_fit, order, action_unit, tune_scale, magnitude_exponent=3):
    scale = (tune_scale * (10 ** -magnitude_exponent)) / (UNIT_TO_M[action_unit] ** order)
    str_val, str_std = _get_scaled_labels(odr_fit.beta[order], odr_fit.sd_beta[order], scale)
    str_mag = ''
    if magnitude_exponent != 0:
        str_mag = f'$\cdot$ 10$^{{{magnitude_exponent}}}$'
    return f'({str_val} $\pm$ {str_std}) {str_mag} m$^{{-{order}}}$'


def _get_odr_label(odr_fit, action_unit, tune_scale):
    order = len(odr_fit.beta) - 1

    str_list = [None] * order
    for o in range(order):
        str_list[o] = _get_scaled_odr_label(odr_fit, o+1, action_unit, tune_scale,
                                            magnitude_exponent=const.get_detuning_exponent_for_order(o+1))
    return ", ".join(str_list)


def plot_odr(ax, odr_fit, xmax, action_unit, tune_scale, color=None):
    """ Adds a quadratic odr fit to axes. """

    color = 'k' if color is None else color
    odr_fit.beta[0] = 0   # no need to remove offset, as it is removed from data
    label = _get_odr_label(odr_fit, action_unit, tune_scale)

    # get fits
    order = len(odr_fit.beta) - 1
    fit_fun = detuning_tools.get_poly_fun(order)
    f = partial(fit_fun, odr_fit.beta)
    f_low = partial(fit_fun, np.array(odr_fit.beta)-np.array(odr_fit.sd_beta))
    f_upp = partial(fit_fun, np.array(odr_fit.beta)+np.array(odr_fit.sd_beta))

    if order == 1:
        x = np.array([0, xmax])
    else:
        x = np.linspace(0, xmax, NFIT)

    ax.fill_between(x, f_low(x), f_upp(x), facecolor=mcolors.to_rgba(color, .3))
    ax.plot(x, f(x), marker="", linestyle='--', color=color, label=label)


# Main Plot ---


def _plot_detuning(ax, data, label, action_unit, tune_scale, color=None, limits=None, odr_fit=None):
    """ Plot the detuning and the ODR into axes. """
    xmax = _get_default(limits, 'xmax', max(data['x']+data['xerr']))

    # Plot Fit
    offset = odr_fit.beta[0]
    plot_odr(ax, odr_fit, xmax=xmax, action_unit=action_unit, tune_scale=tune_scale, color=color)

    # Plot Data
    data['y'] -= offset
    ax.errorbar(**data, label=label, color=color)


def _set_plotstyle(manual_style):
    mstyle = MANUAL_STYLE_DEFAULT
    mstyle.update(manual_style)
    pstyle.set_style("standard", mstyle)


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


def _get_scaled_labels(val, std, scale):
    scaled_vas, scaled_std = val*scale, std*scale
    if abs(scaled_std) > 1 or scaled_std == 0:
        return f'{int(round(scaled_vas)):d}', f'{int(round(scaled_std)):d}'
    return significant_digits(scaled_vas, scaled_std)


# Helper -----------------------------------------------------------------------


def _check_opt(opt):
    if len(opt.labels) != len(opt.kicks):
        raise ValueError("'kicks' and 'labels' need to be of same size!")


def _get_default(ddict, key, default):
    """ Returns 'default' if either the dict itself or the entry is None. """
    if ddict is None or key not in ddict or ddict[key] is None:
        return default
    return ddict[key]


def _correct_and_scale(data, odr_fit, action_unit, action_plot_unit, tune_scale, acd_corr):
    """ Corrects data for AC-Dipole and scales to plot-units (y=tune_scale, x=um)"""
    # scale action units
    x_scale = UNIT_TO_M[action_unit] / UNIT_TO_M[action_plot_unit]
    data['x'] *= x_scale
    data['xerr'] *= x_scale

    # correct for ac-diple and tune scaling
    y_scale = acd_corr / tune_scale
    data['y'] *= y_scale
    data['yerr'] *= y_scale

    # same for odr_fit:
    for idx in range(len(odr_fit.beta)):
        full_scale = y_scale / (x_scale**idx)
        odr_fit.beta[idx] *= full_scale
        odr_fit.sd_beta[idx] *= full_scale
    return data, odr_fit


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
