from pathlib import Path

import pandas as pd
import tfs

from pylhc.irnl_rdt_correction import main as irnl_correct, ORDER_NAME_MAP, BETA, KEYWORD, X, Y, MULTIPOLE

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



def test_correction(tmp_path):
    optics = generate_pseudo_model()
    errors = generate_empty_errortable()
    errors["K3L"] = 1

    _, df_corrections = irnl_correct(
        accel='lhc',
        optics=[optics],
        errors=[errors],
        beams=[1],
        output=tmp_path,
        feddown=0,
        ips=[1, 5],
        ignore_missing_columns=True,
        iterations=1,
    )

    pass


def read_lhc_model(beam):
   return tfs.read_tfs(LHC_MODELS_PATH / f"twiss.lhc.b{beam}.nominal.tfs", index="NAME")


def generate_pseudo_model(betax=1, betay=1, x=0, y=0):
    df = pd.DataFrame(
        index=[f"{name}.{number}{side}{ip}" for ip in range(1, 4) for side in "LR" for name, number in zip("ABC", range(1, 4)) ],
        columns=[f"{BETA}{X}", f"{BETA}{Y}", X, Y, KEYWORD]
    )
    df[f"{BETA}{X}"] = betax
    df[f"{BETA}{Y}"] = betay
    df[X] = x
    df[Y] = y
    df[KEYWORD] = MULTIPOLE
    return df


def generate_empty_errortable():
    df = pd.DataFrame(0,
                      index=[f"{name}.{number}{side}{ip}" for ip in range(1, 4) for side in "LR" for name, number in zip("ABC", range(1, 4)) ],
                      columns=[f"K{n}{o}L" for n in range(6) for o in ("", "S")]
                      )
    return df
