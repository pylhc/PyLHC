"""
Constants: KickGroups
---------------------

Constants used in the KickGroups
"""
from pathlib import Path

KICKGROUPS_ROOT = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat/KickGroups/MULTITURN_ACQ_GROUPS")
KICKGROUP = "KICKGROUP"
SDDS = "SDDS"
TURNS = "TURNS"
BUNCH = "BUNCH"
TIME = "TIME"
TIMESTAMP = "TIMESTAMP"
LOCALTIME = "LOCAL"
TUNEX = "QX"
TUNEY = "QY"
DRIVEN_TUNEX = "DQX"
DRIVEN_TUNEY = "DQY"
DRIVEN_TUNEZ = "DQZ"
AMPX = "AMPX"
AMPY = "AMPY"
AMPZ = "AMPZ"
OPTICS = "OPTICS"
OPTICS_URI = "OPTICS_URI"
BEAMPROCESS = "BEAMPROCESS"
BEAM = "BEAM"
KICK_COLUMNS = [TIME, TUNEX, TUNEY, DRIVEN_TUNEX, DRIVEN_TUNEY, DRIVEN_TUNEZ, AMPX, AMPY, AMPZ, TURNS, BUNCH, SDDS, BEAM, OPTICS, OPTICS_URI, BEAMPROCESS]
COLUMNS_TO_HEADERS = [BEAM, BUNCH, TURNS, BEAMPROCESS, OPTICS, OPTICS_URI]
KICK_GROUP_COLUMNS = [TIME, LOCALTIME, KICKGROUP, TIMESTAMP]