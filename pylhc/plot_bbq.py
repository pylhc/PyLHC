"""
Plotting: BBQ
------------------------------------

Provides the plotting function for the BBQ


:module:  plot_bbq
:author: jdilly

"""
import datetime
from contextlib import suppress

import matplotlib.dates as mdates
import tfs
import numpy as np
from generic_parser import entrypoint, EntryPointParameters
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import FormatStrFormatter


# noinspection PyUnresolvedReferences
from omc3 import amplitude_detuning_analysis as ad_ana
from omc3.tune_analysis import constants as const, kick_file_modifiers as kick_mod
from omc3.utils import logging_tools, plot_style as pstyle
from pylhc.plotshop import colors as pcolors

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
        flags="--in",
        help="BBQ data as data frame or tfs file.",
        name="input",
        required=True,
    )
    params.add_parameter(
        flags="--kick",
        help="Kick file as data frame or tfs file.",
        name="kick",
    )
    params.add_parameter(
        flags="--out",
        help="Save figure to this location.",
        name="output",
        type=str,
    )
    params.add_parameter(
        flags="--show",
        help="Show plot.",
        name="show",
        action="store_true"
    )
    params.add_parameter(
        flags="--xmin",
        help="Lower x-axis limit. (yyyy-mm-dd HH:mm:ss.mmm)",
        name="xmin",
        type=float,
    )
    params.add_parameter(
        flags="--ymin",
        help="Lower y-axis limit.",
        name="ymin",
        type=float,
    )
    params.add_parameter(
        flags="--xmax",
        help="Upper x-axis limit. (yyyy-mm-dd HH:mm:ss.mmm)",
        name="xmax",
        type=float,
    )
    params.add_parameter(
        flags="--ymax",
        help="Upper y-axis limit.",
        name="ymax",
        type=float,
    )
    params.add_parameter(
        flags="--interval",
        help="x_axis interval that was used in calculations.",
        name="interval",
        type=float,
        nargs=2,
    )
    params.add_parameter(
        flags="--two",
        help="Plot two axis into the figure.",
        name="two_plots",
        action="store_true",
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    """ Plot BBQ wrapper.

    Keyword Args:
        Required
        input: BBQ data as data frame or tfs file.
               **Flags**: --in
        Optional
        interval (float): x_axis interval that was used in calculations.
                        **Flags**: --interval
        kick: Kick file as data frame or tfs file (for limiting plotting interval)
              **Flags**: --kick
        output (str): Save figure to this location.
                      **Flags**: --out
        show: Show plot.
              **Flags**: --show
              **Action**: ``store_true``
        two_plots: Plot two axis into the figure.
                   **Flags**: --two
                   **Action**: ``store_true``
        xmax (float): Upper x-axis limit. (yyyy-mm-dd HH:mm:ss.mmm)
                    **Flags**: --xmax
        xmin (float): Lower x-axis limit. (yyyy-mm-dd HH:mm:ss.mmm)
                    **Flags**: --xmin
        ymax (float): Upper y-axis limit.
                      **Flags**: --ymax
        ymin (float): Lower y-axis limit.
                      **Flags**: --ymin
    """
    LOG.info("Plotting BBQ.")
    bbq_df = kick_mod.read_timed_dataframe(opt.input) if isinstance(opt.input, str) else opt.input
    opt.pop("input")

    if opt.kick is not None:
        if opt.interval is not None:
            raise ValueError("interval and kick-file given. Both are used for the same purpose. Please only use one.")

        window = 0  # not too important, bars will then indicate first and last kick directly
        for p in PLANES:
            with suppress(KeyError):
                window = bbq_df.headers[HEADER_MAV_WINDOW(p)]

        kick_df = kick_mod.read_timed_dataframe(opt.kick) if isinstance(opt.kick, str) else opt.kick
        opt.interval = ad_ana.get_approx_bbq_interval(bbq_df, kick_df.index, window)
        bbq_df = bbq_df.loc[opt.interval[0]:opt.interval[1]]
    opt.pop("kick")

    show = opt.pop("show")
    out = opt.pop("output")

    fig = _plot_bbq_data(bbq_df, **opt)

    if show:
        plt.show()
    if out:
        fig.savefig(out)

    plt.close(fig)
    return fig


def _plot_bbq_data(bbq_df,
                   interval=None, xmin=None, xmax=None, ymin=None, ymax=None,
                   two_plots=False):
    """ Plot BBQ data.

    Args:
        bbq_df: BBQ Dataframe with moving average columns
        interval: start and end time of used interval, will be marked with red bars
        xmin: Lower x limit (time)
        xmax: Upper x limit (time)
        ymin: Lower y limit (tune)
        ymax: Upper y limit (tune)
        output: Path to the output file
        show: Shows plot if `True`
        two_plots: Plots each tune in it's own axes if `True`

    Returns:
        Plotted figure

    """
    LOG.debug("Plotting BBQ data.")

    pstyle.set_style("standard", {
        u'figure.figsize': [12.24, 7.68],
        u"lines.marker": u"",
        u"lines.linestyle": u""}
                     )

    fig = plt.figure()

    if two_plots:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax = [fig.add_subplot(gs[1]), fig.add_subplot(gs[0])]
    else:
        gs = gridspec.GridSpec(1, 1, height_ratios=[1])
        ax = fig.add_subplot(gs[0])
        ax = [ax, ax]

    handles = [None] * (3 * len(PLANES))
    for idx, plane in enumerate(PLANES):
        color = pcolors.get_mpl_color(idx)
        mask = np.array(bbq_df[COL_IN_MAV(plane)], dtype=bool)

        # plot and save handles for nicer legend
        handles[idx] = ax[idx].plot([i.get_datetime() for i in bbq_df.index],
                                    bbq_df[COL_BBQ(plane)],
                                    color=pcolors.change_color_brightness(color, .4),
                                    marker="o", markerfacecolor="None",
                                    label="$Q_{:s}$".format(plane.lower(),)
                                    )[0]
        filtered_data = bbq_df.loc[mask, COL_BBQ(plane)].dropna()
        handles[len(PLANES)+idx] = ax[idx].plot(filtered_data.index, filtered_data.values,
                                                color=pcolors.change_color_brightness(color, .7),
                                                marker=".",
                                                label="filtered".format(plane.lower())
                                                )[0]
        handles[2*len(PLANES)+idx] = ax[idx].plot(bbq_df.index, bbq_df[COL_MAV(plane)],
                                                  color=color,
                                                  linestyle="-",
                                                  label="moving av.".format(plane.lower())
                                                  )[0]

        if ymin is None and two_plots:
            ax[idx].set_ylim(bottom=min(bbq_df.loc[mask, COL_BBQ(plane)]))

        if ymax is None and two_plots:
            ax[idx].set_ylim(top=max(bbq_df.loc[mask, COL_BBQ(plane)]))

    # things to add/do only once if there is only one plot
    for idx in range(1+two_plots):
        if interval:
            ax[idx].axvline(x=interval[0], color="red")
            ax[idx].axvline(x=interval[1], color="red")

        if two_plots:
            ax[idx].set_ylabel("$Q_{:s}$".format(PLANES[idx]))
        else:
            ax[idx].set_ylabel('Tune')

        ax[idx].set_ylim(bottom=ymin, top=ymax)
        ax[idx].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))

        ax[idx].set_xlim(left=xmin, right=xmax)
        ax[idx].set_xlabel('Time')
        ax[idx].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        if idx:
            # don't show labels on upper plot (if two plots)
            # use the visibility to allow cursor x-position to be shown
            ax[idx].tick_params(labelbottom=False)
            ax[idx].xaxis.get_label().set_visible(False)

        if not two_plots or idx:
            # reorder legend
            ax[idx].legend(handles, [h.get_label() for h in handles],
                           loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=3,)

    fig.tight_layout()
    fig.tight_layout()
    return fig


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
