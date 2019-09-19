"""
Module plotshop.post_processing
---------------------------------

Functions for plot post-processing.
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np

# Public Functions #############################################################


def merge_two_plots(axes, keep_style=True):
    """ Merges two plots into one.

    Args:
        axes (list): list of axes to merge.
        keep_style (bool): if ``true``, it keeps the lines styles.
    """
    fig = plt.figure()
    new_ax = fig.gca()
    for ax in axes:
        # new_ax.lines += ax.lines
        # new_ax.containers += ax.containers
        lines = get_line_data(ax)
        _plot_collected_data(new_ax, lines, "lines", keep_style)

        ebars = get_errorbar_data(ax)
        _plot_collected_data(new_ax, ebars, "errorbar", keep_style)
    return fig


def transpose_legend(leg):
    """ Transposes the legend. Has some problems with the markers. """
    nrow = leg._ncol
    loc = leg._get_loc()
    handles = np.array(leg.legendHandles)
    ncol = int(np.ceil(len(handles) / float(nrow)))
    order = np.array(range(ncol*nrow)).reshape(nrow, ncol).transpose().ravel()[0:len(handles)]
    handles = handles[order]
    new_leg = leg.axes.legend(handles, [h.get_label() for h in handles], ncol=ncol)
    new_leg._set_loc(loc)
    return new_leg


def sync2d(axs, ax_str='xy', ax_lim=()):
    """
    Synchronizes the limits for the given axes

    Args:
        axs: list of axes or figures, or figure with multiple axes
        ax_lim: predefined limits (list or list of lists)
        ax_str: string 'x','y' or 'xy' defining the axes to sync
    """
    if isinstance(axs, (list, np.ndarray)):
        if isinstance(axs[0], mpl.figure.Figure):
            # axs is list of figures: get all axes and call sync2D
            sync2d([ax for fig in axs for ax in fig.axes], ax_str)

        elif isinstance(axs[0], mpl.axes.Axes) and len(axs) > 1:
            # axs is list of axes
            for idx, axis in enumerate(ax_str):
                get_lim, set_lim = f"get_{axis}lim", f"set_{axis}lim"
                if len(ax_lim) == 0:
                    # find limits
                    all_lim = np.array([getattr(ax, get_lim)() for ax in axs])
                    lim = [np.min(all_lim[:, 0]), np.max(all_lim[:, 1])]
                else:
                    # use defined limits
                    if len(ax_lim[0]) == 1:
                        lim = ax_lim
                    else:
                        lim = ax_lim[idx]

                for ax in axs:
                    getattr(ax, set_lim)(lim)

    elif isinstance(axs, mpl.figure.Figure):
        # axs is one figure: call sync2D with axes from figure
        sync2d(axs.axes, ax_str)

    else:
        raise TypeError(f'{__file__[:-3]}.sync2d input is of unknown type ({str(type(axs))})')


def set_xlimits(accel, ax=None):
    """
    Sets the x-limits to the regularly used ones

    Args:
        accel: Name of the Accelerator
        ax:  Axes to put the label on (default: gca())
    """
    if not ax:
        ax = plt.gca()

    if accel.startswith("LHCB"):
        ax.set_xlim(-200, 27000)
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=1000.0))
        ax.xaxis.set_major_locator(mtick.MultipleLocator(base=5000.0))
    elif accel.startswith("ESRF"):
        ax.set_xlim(-5, 850)
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=50.0))
        ax.xaxis.set_major_locator(mtick.MultipleLocator(base=100.0))
    else:
        raise ValueError(f"Accelerator '{accel}' unknown.")



# Data Extraction ##############################################################


def get_errorbar_data(ax):
    """ Extract data from all errorbars in axes.

    Args:
        ax: axes handle to axes to extract data from.

    Returns:
        :List of dictionaries of extracted data with keys as follows:

        |    label: Line label
        |    x: x-position data of the points
        |    y: y-position data of the points
        |    xerr: tupel with arrays corresponding to lower and upper x-error values
        |    yerr: tupel with arrays corresponding to lower and upper y-error values
        |    with: Linewidth
        |    style: Linestyle
        |    color: Linecolor

        For ``"_nolegend_"`` entries an index is added to avoid collision.

    See Also:
        ``get_line_data()``

    """
    data = []
    for idx, ebar in enumerate(ax.containers):
        if isinstance(ebar, mpl.container.ErrorbarContainer):
            line_dict = _extract_line_data(ebar[0])
            line_dict.update({
                "label": ebar.get_label(),
                "xerr": _get_ebar_err(ebar, "x"),
                "yerr": _get_ebar_err(ebar, "y"),
            })
            data.append(line_dict)
    return data


def get_line_data(ax):
    """ Extract data from all lines in axes.

    Args:
        ax: axes handle to axes to extract data from.

    Returns:
        :List of dictionaries of extracted data with keys as follows:

        |    label: Line label
        |    x: x-position data of the points
        |    y: y-position data of the points
        |    with: Linewidth
        |    style: Linestyle
        |    color: Linecolor

        If a line is found to be an error-bar-line it is ignored!
        For ``"_nolegend_"`` entries an index is added to avoid collision.

    See Also:
        ``get_errorbar_data()``

    """
    ax_errorbars = [ebar[0] for ebar in ax.containers
                    if isinstance(ebar, mpl.container.ErrorbarContainer)]
    data = []
    for idx, line in enumerate(ax.get_lines()):
        if line not in ax_errorbars:
            data.append(_extract_line_data(line))
    return data


# Private Functions ############################################################


def _get_ebar_err(ebar, plane):
    """ Extract the error information from the errorbar. """
    data = ebar[0].get_xdata() if plane == "x" else ebar[0].get_ydata()
    plane_idx = 0 if plane == "x" else 1

    segments = ebar[2][plane_idx].get_segments()

    lower = np.zeros(len(data))
    upper = np.zeros(len(data))
    for idx, (dat, bar) in enumerate(zip(data, segments)):
        lower[idx] = dat - bar[0][plane_idx]
        upper[idx] = bar[1][plane_idx] - dat
    return lower, upper


def _plot_collected_data(ax, data, data_type, keep_style):
    plot_fun = {
        "errorbar": ax.errorbar,
        "lines": lambda **kwargs: ax.plot(kwargs.pop("x"), kwargs.pop("y"), **kwargs),  # mpl !!
    }[data_type]

    for dat in data:
        if not keep_style:
            [dat.pop(key) for key in ["linewidth", "linestyle", "color", "markersize"]]
        plot_fun(**dat)


def _extract_line_data(line):
        return {
            "label": line.get_label(),
            "x": line.get_xdata(),
            "y": line.get_ydata(),
            "linewidth": line.get_linewidth(),
            "linestyle": line.get_linestyle(),
            "color": line.get_color(),
            "marker": line.get_marker(),
            "markersize": line.get_markersize(),
        }


