import tfs
import pandas as pd 

ELEMENTSTYLE = {
    'QUADRUPOLE': {'color': 'royalblue'},
    'SBEND': {'color': 'darkorange'},
    'RBEND': {'color': 'darkorange'}
}


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


def plot_layout(ax, layoutdf, plot_dict):
    ax.axhline(color='black')
    ax.set_ylim([-1.2, 1.2])

    for idx, elem in layoutdf.iterrows():
        try:
            ax.bar(elem['S']-elem['L']/2., get_height(elem), elem['L'],
                   bottom=get_bottom(elem),
                   color=ELEMENTSTYLE[elem['KEYWORD']]['color'],
                   alpha=plot_dict['alpha'])
        except KeyError:
            pass