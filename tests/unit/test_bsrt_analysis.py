import os
from pathlib import Path
from pylhc import BSRT_analysis

def test_bsrt_analysis(file_regression):
    BSRT_analysis.main(directory=os.path.join('..', 'inputs', 'bsrt_analysis'), beam='B1', outputdir='../outputs/')
    file_regression.check(str(Path('../outputs/', BSRT_analysis._get_bsrt_tfs_fname('B1'))))
    file_regression.check(str(Path('../outputs/', BSRT_analysis._get_fitvar_plot_fname('B1'))))
    file_regression.check(str(Path('../outputs/', BSRT_analysis._get_2dcrossection_plot_fname('B1'))))
    file_regression.check(str(Path('../outputs/', BSRT_analysis._get_auxiliary_var_plot_fname('B1'))), extension='.pdf')
