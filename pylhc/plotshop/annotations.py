import re
from distutils.version import LooseVersion

import matplotlib
import pandas as pd
import tfs
from matplotlib import pyplot as plt

from plotshop.style import ArgumentError

# List of common y-labels. Sorry for the ugly.
_ylabels = {
    "beta":               r'$\beta_{{{0}}} \quad [m]$',
    "betabeat":           r'$\Delta \beta_{{{0}}} / \beta_{{{0}}}$',
    "betabeat_permile":   r'$\Delta \beta_{{{0}}} / \beta_{{{0}}} [$'u'\u2030'r'$]$',
    "dbeta":              r"$\beta'_{{{0}}} \quad [m]$",
    "dbetabeat":          r'$1/\beta_{{{0}}} \cdot \partial\beta_{{{0}}} / \partial\delta_{{{0}}}$',
    "norm_dispersion":    r'$\frac{{D_{{{0}}}}}{{\sqrt{{\beta_{{{0}}}}}}} \quad [\sqrt{{m}}]$',
    "norm_dispersion_mu": r'$\frac{{D_{{{0}}}}}{{\sqrt{{\beta_{{{0}}}}}}} \quad [\mu \sqrt{{m}}]$',
    "phase":              r'$\phi_{{{0}}} \quad [2\pi]$',
    "phasetot":           r'$\phi_{{{0}}} \quad [2\pi]$',
    "phase_milli":        r'$\phi_{{{0}}} \quad [2\pi\cdot10^{{-3}}]$',
    "dispersion":         r'$D_{{{0}}} \quad [m]$',
    "dispersion_mm":      r'$D_{{{0}}} \quad [mm]$',
    "co":                 r'${0} \quad [mm]$',
    "tune":               r'$Q_{{{0}}} \quad [Hz]$',
    "nattune":            r'$Nat Q_{{{0}}} \quad [Hz]$',
    "chromamp":           r'$W_{{{0}}}$',
    "real":               r'$re({0})$',
    "imag":               r'$im({0})$',
    "absolute":           r'$|{0}|$',
}


def set_yaxis_label(param, plane, ax=None, delta=False, chromcoup=False):  # plot x and plot y
    """ Set y-axis labels.

    Args:
        param: One of the ylabels above
        plane: Usually x or y, but can be any string actually to be placed into the label ({0})
        ax: Axes to put the label on (default: gca())
        delta: If True adds a Delta before the label (default: False)
    """
    if not ax:
        ax = plt.gca()
    try:
        label = _ylabels[param].format(plane)
    except KeyError:
        raise ArgumentError("Label '" + param + "' not found.")

    if delta:
        if param.startswith("beta") or param.startswith("norm"):
            label = r'$\Delta(' + label[1:-1] + ")$"
        else:
            label = r'$\Delta ' + label[1:]

    if chromcoup:
        label = label[:-1] + r'/\Delta\delta$'

    ax.set_ylabel(label)


def set_xaxis_label(ax=None):
    """ Sets the standard x-axis label

    Args:
        ax: Axes to put the label on (default: gca())
    """
    if not ax:
        ax = plt.gca()
    ax.set_xlabel(r'Longitudinal location [m]')


def show_ir(ip_dict, ax=None, mode='inside'):
    """ Plots the interaction regions into the background of the plot.

    Args:
        ip_dict: dict, dataframe or series containing "IPLABEL" : IP_POSITION
        ax:  Axes to put the irs on (default: gca())
        mode: 'inside', 'outside' + 'nolines' or just 'lines'
    """
    if not ax:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lines = 'nolines' not in mode
    inside = 'inside' in mode
    lines_only = 'inside' not in mode and 'outside' not in mode and 'lines' in mode

    if isinstance(ip_dict, (pd.DataFrame, pd.Series)):
        if isinstance(ip_dict, pd.DataFrame):
            ip_dict = ip_dict.iloc[:, 0]
        d = {}
        for ip in ip_dict.index:
            d[ip] = ip_dict.loc[ip]
        ip_dict = d

    for ip in ip_dict.keys():
        if xlim[0] <= ip_dict[ip] <= xlim[1]:
            xpos = ip_dict[ip]

            if lines:
                ax.axvline(xpos, linestyle=':', color='grey', marker='', zorder=0)

            if not lines_only:
                ypos = ylim[not inside] + (ylim[1] - ylim[0]) * 0.01
                c = 'grey' if inside else matplotlib.rcParams["text.color"]
                ax.text(xpos, ypos, ip, color=c, ha='center', va='bottom')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def move_ip_labels(ax, value):
    """ Moves IP labels according to max y * value."""
    y_max = ax.get_ylim()[1]
    for t in ax.texts:
        if re.match(r"^IP\s*\d$", t.get_text()):
            x = t.get_position()[0]
            t.set_position((x, y_max * value))


def get_ip_positions(path):
    """ Returns a dict of IP positions from tfs-file of path.

    Args:
        path (str): Path to the tfs-file containing IP-positions
    """
    df = tfs.read_tfs(path).set_index('NAME')
    ip_names = ["IP" + str(i) for i in range(1, 9)]
    ip_pos = df.loc[ip_names, 'S'].values
    return dict(zip(ip_names, ip_pos))


def set_name(name, fig_or_ax=None):
    """ Sets the name of the figure or axes

    Args:
        name (str): Sting to set as name.
        fig_or_ax: Figure or Axes to to use.
            If 'None' takes current figure. (Default: None)
    """
    if not fig_or_ax:
        fig_or_ax = plt.gcf()
    try:
        fig_or_ax.figure.canvas.set_window_title(name)
    except AttributeError:
        fig_or_ax.canvas.set_window_title(name)


def get_name(fig_or_ax=None):
    """ Returns the name of the figure or axes

    Args:
        fig_or_ax: Figure or Axes to to use.
            If 'None' takes current figure. (Default: None)
    """
    if not fig_or_ax:
        fig_or_ax = plt.gcf()
    try:
        return fig_or_ax.figure.canvas.get_window_title()
    except AttributeError:
        return fig_or_ax.canvas.get_window_title()


def set_annotation(text, ax=None, position='right'):
    """ Writes an annotation on the top right of the axes

    Args:
        text: The annotation
        ax: Axes to set annotation on. If 'None' takes current Axes. (Default: None)
    """
    if not ax:
        ax = plt.gca()

    annotation = get_annotation(ax, by_reference=True)

    xpos = float(position == 'right')

    if annotation is None:
        ax.text(xpos, 1.0, text,
                ha=position,
                va='bottom',
                transform=ax.transAxes,
                label='plot_style_annotation')
    else:
        annotation.set_text(text)
        plt.setp(annotation, ha=position, x=xpos)


def get_annotation(ax=None, by_reference=True):
    """ Returns the annotation set by set_annotation()

    Args:
        ax: Axes to get annotation from. If 'None' takes current Axes. (Default: None)
        by_reference (bool): If true returns the reference to the annotation,
            otherwise the text as string. (Default: True)
    """
    if not ax:
        ax = plt.gca()

    for c in ax.get_children():
        if c.get_label() == 'plot_style_annotation':
            if by_reference:
                return c
            else:
                return c.get_text()
    return None


def small_title(ax=None):
    """ Alternative to annotation, which lets you use the title-functions

    Args:
        ax: Axes to use. If 'None' takes current Axes. (Default: None)
    """
    if not ax:
        ax = plt.gca()

    # could not get set_title() to work properly, so one parameter at a time
    ax.title.set_position([1.0, 1.02])
    ax.title.set_transform(ax.transAxes)
    ax.title.set_fontsize(matplotlib.rcParams['font.size'])
    ax.title.set_fontweight(matplotlib.rcParams['font.weight'])
    ax.title.set_verticalalignment('bottom')
    ax.title.set_horizontalalignment('right')


def figure_title(text, ax=None, pad=0, **kwargs):
    """ Set the title all the way to the top.

    Args:
        text: Text for the title
        ax: Axes to use. If 'None' takes current Axes. (Default: None)
        pad: Padding from border
        kwargs: passed on to fontdict
    """
    if not ax:
        ax = plt.gca()

    # could not get set_title() to work properly, so one parameter at a time
    fdict = dict(fontsize=matplotlib.rcParams['font.size'],
                 fontweight=matplotlib.rcParams['font.weight'],
                 va="top", ha="center" )
    fdict.update(kwargs)
    ax.set_title(text, transform=ax.figure.transFigure, fontdict=fdict)

    ax.title.set_position([.5, float(fdict['va'] == "top") + pad * (-1)**(fdict['va'] == 'top')])


def get_legend_ncols(labels, max_length=78):
    """ Calulate the number of columns in legend dynamically """
    return max([max_length/max([len(l) for l in labels]), 1])


def make_top_legend(ax, ncol, frame=False, handles=None, labels=None):
    """ Create a legend on top of the plot. """
    leg = ax.legend(handles=handles, labels=labels, loc='lower right',
                    bbox_to_anchor=(1.0, 1.02),
                    fancybox=frame, shadow=frame, frameon=frame, ncol=ncol)

    if LooseVersion(matplotlib.__version__) <= LooseVersion("2.2.0"):
        legend_height = leg.get_window_extent().inverse_transformed(leg.axes.transAxes).height
        ax.figure.tight_layout(rect=[0, 0, 1, 1-legend_height])

    leg.axes.figure.canvas.draw()
    legend_width = leg.get_window_extent().inverse_transformed(leg.axes.transAxes).width
    if legend_width > 1:
        x_shift = (legend_width - 1) / 2.
        ax.legend(handles=handles, labels=labels, loc='lower right',
                  bbox_to_anchor=(1.0 + x_shift, 1.02),
                  fancybox=frame, shadow=frame, frameon=frame, ncol=ncol)

    if LooseVersion(matplotlib.__version__) >= LooseVersion("2.2.0"):
        ax.figure.tight_layout()

    return leg


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    """ Order of Magnitude Formatter.

    To set a fixed order of magnitude and fixed significant numbers.
    As seen on: https://stackoverflow.com/a/42658124/5609590

    See: set_sci_magnitude

    """
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


def set_sci_magnitude(ax, axis="both", order=0, fformat="%1.1f", offset=True, math_text=True):
    """ Uses the OMMFormatter to set the scientific limits on axes.

    Args:
        ax: Plotting axes
        axis (str): "x", "y" or "both"
        order (int): Magnitude Order
        fformat (str): Format to use
        offset (bool): Formatter offset
        math_text (bool): Whether to use mathText
    """
    oomf = OOMFormatter(order=order, fformat=fformat, offset=offset, mathText=math_text)

    if axis == "x" or axis == "both":
        ax.xaxis.set_major_formatter(oomf)

    if axis == "y" or axis == "both":
        ax.yaxis.set_major_formatter(oomf)

    ax.ticklabel_format(axis=axis, style="sci", scilimits=(order, order), useMathText=math_text)

    return ax