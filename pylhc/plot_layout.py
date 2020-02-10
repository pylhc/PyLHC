import numpy as np
import tfs
import pandas as pd
import matplotlib.pyplot as plt
from generic_parser import entrypoint, EntryPointParameters


ELEMENTSTYLE = {
    'QUADRUPOLE': {'color': 'royalblue'},
    'SBEND': {'color': 'darkorange'},
    'RBEND': {'color': 'darkorange'}
}

LOG = logging_tools.get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="layout_twiss",
        type=str,
        required=True,
        help="Twiss file containing layout to be plotted. Necessary columns are NAME, S, KEYWORD",
    )
    params.add_parameter(
        name="optics_twiss",
        type=str,
        required=True,
        help="Twiss file containing optics to be plotted. Necessary columns are NAME, S, BETX, BETY",
    )
    params.add_parameter(
        name="outputfile",
        type=str,
        required=True,
        help="Outfilename",
    )

    return params

@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.INFO('Firing up the plotter!')
    layout=tfs.read(opt.layout_twiss, index='NAME')
    optics=tfs.read(opt.optics_twiss, index='NAME')

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,9), gridspec_kw={'height_ratios':[1,3]})

    axlayout = ax[0]
    axopt = ax[1]

    add_magnets(axlayout, layout)

    add_optics(axopt, optics, 'IP1')

    plt.tight_layout()
    plt.savefig(opt.outputfile)
    plt.show()



def get_height(elem):
    if elem['KEYWORD'] == 'QUADRUPOLE':
        return elem['K1L']/np.abs(elem['K1L'])
    elif elem['KEYWORD'] == 'SBEND':
        return 1
    elif elem['KEYWORD'] == 'RBEND':
        return 1


def get_bottom(elem):
    if elem['KEYWORD'] == 'SBEND':
        return -0.5
    if elem['KEYWORD'] == 'RBEND':
        return -0.5
    else:
        return 0


def add_magnets(ax, data):

    ax.axhline(color='black')
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim([-1.2, 1.2])

    for idx, elem in data.iterrows():
        try:
            ax.bar(elem['S']-elem['L']/2., get_height(elem), elem['L'],
                   bottom=get_bottom(elem),
                   color=ELEMENTSTYLE[elem['KEYWORD']]['color'],
                   alpha=0.8)
        except KeyError:
            pass


def add_optics(ax, data, center):
    ax.plot(data['S']-data.loc[center, 'S'], data['BETX'], label=r'$\beta_x$', color='crimson', linewidth=3)
    ax.plot(data['S']-data.loc[center, 'S'], data['BETY'], label=r'$\beta_y$', color='mediumblue', linewidth=3)
    ax.grid(True)
    ax.set_xlabel('S [m]', fontsize=20)
    ax.set_ylabel(r'$\beta_{x,y}~[m]$', fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelsize=20)

