"""
BSRT logger
-------------------

Script used during Run II to log detailed BSRT data and save it for later analysis. Data from the
BSRT for each timestep is put in a `dictionary` and append to a `list`. The `list` is then saved
to disk through pickling. Proper testing requires communication with ``FESA`` s class, possible
only from the Technical Network.

Original authors: E. H. Maclean, T. Persson and G. Trad.
"""
import datetime as dt
import os
import pickle
import sys
import time

from omc3.definitions import formats
from omc3.utils.mock import cern_network_import

pyjapc = cern_network_import("pyjapc")


##########################################


def parse_timestamp(thistime):
    accepted_time_input_format = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d_%H:%M:%S.%f",
        "%Y-%m-%d_%H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
    ]
    for fmat in accepted_time_input_format:
        try:
            dtobject = dt.datetime.strptime(thistime, fmat)
            return dtobject
        except ValueError:
            pass
    timefmatstring = ""
    for fmat in accepted_time_input_format:
        timefmatstring = timefmatstring + '"' + fmat + '" ,   '
    sys.tracebacklimit = 0
    raise ValueError(
        "No appropriate input format found for start time of scan (-s).\n "
        "---> Accepted input formats are:   " + timefmatstring
    )


##########################################


# function to help write output from datetime objects in standard format throughout code
def convert_to_data_output_format(dtobject):
    output_timestamp = dtobject.strftime(formats.TIME)
    return output_timestamp


##########################################


if __name__ == '__main__':
    # Create a PyJapc instance with selector SCT.USER.ALL
    # INCA is automatically configured based on the timing domain you specify here

    CycleName = "LHC.USER.ALL"
    INCAacc = "LHC"
    noSetFlag = True

    japc = pyjapc.PyJapc(selector=CycleName, incaAcceleratorName=INCAacc, noSet=noSetFlag)
    japc.rbacLogin()
    acquesitions_per_file = 100
    j = 0
    t = 0
    while True:
        time.sleep(0.7)
        print(j, t)
        B1_image = japc.getParam("LHC.BSRTS.5R4.B1/Image")
        B2_image = japc.getParam("LHC.BSRTS.5L4.B2/Image")
        if t == 0:
            allB1data = []
            allB2data = []
            B1_IMGtime = B1_image["acqTime"]
            B2_IMGtime = B2_image["acqTime"]
            B1_IMGtime_dt = parse_timestamp(B1_IMGtime)
            B2_IMGtime_dt = parse_timestamp(B2_IMGtime)
            B1_IMGtime_st = convert_to_data_output_format(B1_IMGtime_dt)
            B2_IMGtime_st = convert_to_data_output_format(B2_IMGtime_dt)

        allB1data.append(B1_image)
        allB2data.append(B2_image)
        t += 1
        if t == acquesitions_per_file:
            j += 1
            f1name = "data_BSRT_B1_" + B1_IMGtime_st + ".dat"
            f2name = "data_BSRT_B2_" + B2_IMGtime_st + ".dat"
            f1 = open(f1name, "wb")
            f2 = open(f2name, "wb")
            pickle.dump(allB1data, f1)
            pickle.dump(allB2data, f2)
            f1.close()
            f2.close()
            os.system("gzip " + f1name)
            os.system("gzip " + f2name)
            t = 0

    # Close the RBAC session
    japc.rbacLogout()  # this code in not reachable! Encapsulate in a try..except maybe?
