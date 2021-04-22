"""
Constants: BPM Calibration
--------------------------

Specific constants related to the BPM calibration script in ``PyLHC``.
"""

# IPS to be used to compute the calibraton factors
IPS = [1, 4, 5]

# Constants for TFS files
LABELS = ['S',
          'CALIBRATION',
          'ERROR_CALIBRATION',
          'CALIBRATION_FIT',
          'ERROR_CALIBRATION_FIT']
TFS_INDEX = 'NAME'
D = 'D'
ND = 'ND'

# Estimation for the curve fit
BETA_STAR_ESTIMATION = 200

# Methods to be used to compulte the calibration factors
METHODS = ('beta', 'dispersion')

# File name prefix for calibration output
# end result example: {'beta': 'calibration_beta_.tfs', 'dispersion' ... }
CALIBRATION_NAME = {m: f'calibration_{m}_' for m in METHODS}

# Define BPMs to be used for a combination of IP and Beam
BPMS = {1: {1: ['BPMR.5L1.B1',
                'BPMYA.4L1.B1',
                'BPMWB.4L1.B1',
                'BPMSY.4L1.B1',
                'BPMS.2L1.B1',
                'BPMSW.1L1.B1',
                'BPMSW.1R1.B1',
                'BPMS.2R1.B1',
                'BPMSY.4R1.B1',
                'BPMWB.4R1.B1',
                'BPMYA.4R1.B1'],
            2: ['BPM.5L1.B2', 
                'BPMYA.4L1.B2', 
                'BPMWB.4L1.B2', 
                'BPMSY.4L1.B2', 
                'BPMS.2L1.B2', 
                'BPMSW.1L1.B2', 
                'BPMSW.1R1.B2', 
                'BPMS.2R1.B2', 
                'BPMSY.4R1.B2', 
                'BPMWB.4R1.B2', 
                'BPMYA.4R1.B2']
            },
        4: {1: [
                'BPMYA.5L4.B1',
                'BPMWI.A5L4.B1',
                'BPMWA.B5L4.B1',
                'BPMWA.A5L4.B1',
                'BPMWA.A5R4.B1',
                'BPMWA.B5R4.B1',
                'BPMYB.5R4.B1',
                'BPMYA.6R4.B1',
                ],
            2: [ 
                'BPMYB.5L4.B2',
                'BPMWA.B5L4.B2',
                'BPMWA.A5L4.B2',
                'BPMWA.A5R4.B2',
                'BPMWA.B5R4.B2',
                'BPMWI.A5R4.B2',
                'BPMYA.5R4.B2',
                'BPMYB.6R4.B2'
                ]
             },
        5: {1: ['BPMYA.4L5.B1', 
                'BPMWB.4L5.B1', 
                'BPMSY.4L5.B1', 
                'BPMS.2L5.B1', 
                'BPMSW.1L5.B1', 
                'BPMSW.1R5.B1', 
                'BPMS.2R5.B1', 
                'BPMSY.4R5.B1',
                'BPMWB.4R5.B1', 
                'BPMYA.4R5.B1', 
                'BPM.5R5.B1'],
            2: ['BPMYA.4L5.B2', 
                'BPMWB.4L5.B2', 
                'BPMSY.4L5.B2', 
                'BPMS.2L5.B2', 
                'BPMSW.1L5.B2', 
                'BPMSW.1R5.B2', 
                'BPMS.2R5.B2', 
                'BPMSY.4R5.B2', 
                'BPMWB.4R5.B2', 
                'BPMYA.4R5.B2', 
                'BPMR.5R5.B2']
            }
        }

# For the dispersion method, only a subject of the BPMs is used
# Same as BPM: IP and then beam
D_BPMS = {1: {1: ['BPMSY.4L1.B1',
                  'BPMS.2L1.B1',
                  'BPMSW.1L1.B1',
                  'BPMSW.1R1.B1',
                  'BPMS.2R1.B1',
                  'BPMSY.4R1.B1'],
              2: ['BPMSY.4L1.B2', 
                  'BPMS.2L1.B2', 
                  'BPMSW.1L1.B2', 
                  'BPMSW.1R1.B2', 
                  'BPMS.2R1.B2', 
                  'BPMSY.4R1.B2']
              },
          4: {1: [],
              2: []
              },
          5: {1: ['BPMSY.4L5.B1', 
                  'BPMS.2L5.B1', 
                  'BPMSW.1L5.B1', 
                  'BPMSW.1R5.B1', 
                  'BPMS.2R5.B1', 
                  'BPMSY.4R5.B1',
                  ],
              2: ['BPMSY.4L5.B2', 
                  'BPMS.2L5.B2', 
                  'BPMSW.1L5.B2', 
                  'BPMSW.1R5.B2', 
                  'BPMS.2R5.B2', 
                  'BPMSY.4R5.B2']
              }
          }
