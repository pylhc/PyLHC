import matplotlib.pyplot as plt 
import tfs
import pandas as pd 


def plot_beta(ax, opticsdf, plot_dict):

    if plot_dict['center'] not None:
        plotdf = opticsdf['S']-opticsdf.loc[center, 'S']
    else:
        plotdf = opticsdf

    ax.plot(plotdf['S'], opticsdf['BETX'], label=r'$\beta_x$', color='crimson', linewidth=3)
    ax.plot(plotdf['S'], opticsdf['BETY'], label=r'$\beta_y$', color='mediumblue', linewidth=3)
    ax.set_xlabel('S [m]', fontsize=plot_dict['labelsize'])
    ax.set_ylabel(r'$\beta_{x,y}~[m]$', fontsize=plot_dict['labelsize'])
    ax.legend(fontsize=plot_dict['legendsize'])
