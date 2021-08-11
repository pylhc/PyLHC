from pathlib import Path

import pandas as pd
import tfs

from pylhc.irnl_rdt_correction import main as irnl_correct, ORDER_NAME_MAP

INPUTS = Path(__file__).parent.parent / "inputs"
LHC_MODELS_PATH = INPUTS / "model_lhc_thin_30cm"

# irnl_correct(
#     accel=,
#     optics=,
#     errors=,
#     beams=,
#     output=,
#     rdts=,
#     feddown=,
#     ips=,
#     ignore_missing_columns=,
#     iterations=1,
# )
#

def test_lhc_correction(tmp_path, beam, order, feeddown):
    optics = read_lhc_model(beam)
    errors = pd.DataFrame(["MQX."])

    irnl_correct(
        accel='lhc',
        optics=optics,
        errors=,
        beams=[beam],
        output=tmp_path,
        feddown=feeddown,
        ips=[1,5],
        ignore_missing_columns=True,
        iterations=1,
    )


def read_lhc_model(beam):
   return tfs.read_tfs(LHC_MODELS_PATH / f"twiss.lhc.b{beam}.nominal.tfs", index="NAME")
