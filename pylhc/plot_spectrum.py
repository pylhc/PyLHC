"""
Spectrum Plotter
--------------------

Takes data from frequency analysis and creates a frequency plot for every given BPM,
with the possibility to include spectral lines.
Optionally, a waterfall plot for all BPMs is created as well.
Plots are saved in a directory with the name of the original TbT file.
Returns a dict with the BPM as key and the figure as value for further processing.

required arguments:
    --files                 List with basenames of Tbt files, ie. tracking.sdds
    --working_directory     Directory containing the Frequency files, amplitude, etc.
    --harpy_directory       Directory to write results to

optional arguments:
    --bpms                  List of BPMs for which spectra will be plotted
    --waterfall_plot        Flag to create waterfall plot
    --no_plots              Flag to stopped saving plots
    --lines                 Dict of lines to plot, key being label, value a list with order, ie. {"3Q_x":[2,0]}

"""
import os
import tfs
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters, save_options_to_config
from generic_parser.entry_datatypes import DictAsString

PLANES = {'x': 0, 'y': 1}


def spectrum_plot_entrypoint():
    params = EntryPointParameters()
    params.add_parameter(name="files",
                         required=True,
                         nargs='+',
                         help='List with basenames of Tbt files, ie. tracking.sdds')
    params.add_parameter(name="harpy_directory",
                         required=True,
                         help='Directory containing the Frequency files, amplitude, etc.')
    params.add_parameter(name="working_directory",
                         required=True,
                         help='Directory to write results to')
    params.add_parameter(name="bpms",
                         nargs='+',
                         default=[None],
                         help='List of BPMs for which spectra will be plotted')
    params.add_parameter(name="waterfall_plot",
                         action="store_true",
                         help='Flag to create waterfall plot')
    params.add_parameter(name="return__plots",
                         action="store_true",
                         help='Flag to return dict of plots')
    params.add_parameter(name="no_plots",
                         action="store_true",
                         help='Flag to stopped saving plots')
    params.add_parameter(name="lines",
                         type=DictAsString,
                         default={'Q_x': [1, 0], 'Q_y': [0, 1]},
                         help='Dict of lines to plot, key being label, value a list with order, ie. {"3Q_x":[2,0]}')
    return params


def get_amplitude_files(cwd, kickfile):
    return {plane: tfs.read(f'{cwd}/{kickfile}.amps{plane}') for plane in PLANES.keys()}


def get_frequency_files(cwd, kickfile):
    return {plane: tfs.read(f'{cwd}/{kickfile}.freqs{plane}') for plane in PLANES.keys()}


def get_lin_files(cwd, kickfile):
    return {plane: tfs.read(f'{cwd}/{kickfile}.lin{plane}', index='NAME') for plane in PLANES.keys()}


def get_tune(lin, bpm=None):
    if bpm is None:
        return {f'Q{plane}': lin[plane][f'TUNE{plane.upper()}'].mean() for plane in PLANES.keys()}
    else:
        return {f'Q{plane}': lin[plane].loc[bpm, f'TUNE{plane.upper()}'] for plane in PLANES.keys()}


def plot_stems(ax, freq, amp, bpm):
    ax.stem(freq[bpm],
            amp[bpm],
            use_line_collection=True,
            linefmt='bo-',
            markerfmt='bo',
            basefmt='b-')
    ax.set_yscale('log')
    ax.set_ylim([10**-9, 10**-3])
    ax.set_ylabel('Amplitude [a.u]', fontsize=15)
    ax.set_xlabel('Frequency [tune units]', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc='upper right', frameon=True, fontsize=12)


def plot_line(ax, label, resonance, tunes):
    ax.axvline(x=[np.mod(resonance@tunes, 1) if np.mod(resonance@tunes, 1) < 0.5 else 1-np.mod(resonance@tunes, 1)],
               label=rf'${label}$',
               linestyle='--',
               color='black')


def save_plot(fname):
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    plt.close()


def create_stem_plots(cwd, amp, freq, lin, opt):

    common_bpms = [bpm for bpm in list(lin['x'].index.intersection(lin['y'].index)) if bpm in opt.bpms]
    bpm_figs = {}
    for bpm in common_bpms:
        tune = get_tune(lin, bpm)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))
        for plane, value in PLANES.items():
            plot_stems(ax[value], freq[plane], amp[plane], bpm)
            if opt.lines is not None:
                for label, resonance in opt.lines.items():
                    plot_line(ax[value], label, np.array(resonance), np.array([*tune.values()]))

        if not opt.no_plots:
            save_plot(f'{cwd}/{bpm}_spectrum.png')
        if opt.return_plots:
            bpm_figs[bpm] = fig

    return bpm_figs


def rescale_amp(amp, plane):
    return amp[plane].divide(amp[plane].max(axis=0), axis=1)


def plot_waterfall(ax, freq, amp, bpm_indices, bpm_to_index):
    ax.scatter(x=freq,
               y=bpm_indices,
               c=amp,
               s=250,
               cmap='inferno',
               marker='|',
               alpha=0.5,
               norm=matplotlib.colors.LogNorm())
    ax.set_yticks([*bpm_to_index.values()])
    ax.set_yticklabels(bpm_to_index.keys(), fontdict={'fontsize': 6})
    ax.set_xlabel('Frequency [tune units]', fontsize=15)
    ax.set_ylabel('BPM', fontsize=15)
    ax.set_xlim([0, 0.5])
    ax.tick_params(axis='x', which='both', labelsize=15)


def create_waterfall_plot(cwd, amp, freq, lin, opt):
    amp_stacked = {plane: rescale_amp(amp, plane) for plane in PLANES.keys()}
    amp_stacked = {plane: np.vstack(amp_stacked[plane].to_numpy().T) for plane in PLANES.keys()}
    freq_stacked = {plane: np.vstack(freq[plane].to_numpy().T) for plane in PLANES.keys()}
    bpm_indices = {plane: (np.ones(np.shape(freq[plane].to_numpy()))*np.arange(1, 1+np.shape(freq[plane].to_numpy())[1])).T.flatten() for plane in PLANES.keys()}  # good luck understanding this one

    bpm_to_index = {bpm: i+1 for i, bpm in enumerate(amp['x'].columns)}

    tune = get_tune(lin)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))
    for plane, value in PLANES.items():
        plot_waterfall(ax[value], freq_stacked[plane], amp_stacked[plane], bpm_indices[plane], bpm_to_index)
        if opt.lines is not None:
            for label, resonance in opt.lines.items():
                plot_line(ax[value], label, np.array(resonance), np.array([*tune.values()]))
    if not opt.no_plots:
        save_plot(f'{cwd}/waterfall_spectrum.png')
    if opt.return_plots:
        return fig


def make_cwd(opt, kickfile):
    cwd = os.path.join(opt.working_directory, os.path.splitext(kickfile)[0])
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    return cwd


@entrypoint(spectrum_plot_entrypoint(), strict=True)
def spectrum_plots(opt):

    save_options_to_config(os.path.join(opt.working_directory, 'plot_spectrum.ini'), OrderedDict(sorted(opt.items())))
    opt.files = [os.path.basename(filename) for filename in opt.files]

    bpm_figs = {}
    waterfall_figs = {}

    for kickfile in opt.files:

        cwd = make_cwd(opt, kickfile)

        amp = get_amplitude_files(opt.harpy_directory, kickfile)
        freq = get_frequency_files(opt.harpy_directory, kickfile)
        lin = get_lin_files(opt.harpy_directory, kickfile)

        bpm_fig = create_stem_plots(cwd, amp, freq, lin, opt)

        if opt.waterfall_plot:
            waterfall_fig = create_waterfall_plot(cwd, amp, freq, lin, opt)
        else:
            waterfall_fig = None

        bpm_figs[kickfile] = bpm_fig
        waterfall_figs[kickfile] = waterfall_fig

    return bpm_figs, waterfall_figs


if __name__ == "__main__":
    spectrum_plots()
