import sys
from os.path import abspath, join, dirname
omc3_path = abspath(join(dirname(__file__), "omc3", "omc3"))
if omc3_path not in sys.path:
    sys.path.append(omc3_path)