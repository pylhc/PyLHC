"""
Kickgroups
----------

Functions to list KickGroups and show their Kicks.

.. code-block:: none

    usage: kickgroups.py [-h] {kickgroups,kickgroup_info} ...

    KickGroups Functions

    optional arguments:
      -h, --help            show this help message and exit

    Functionality:
      {kickgroups,kickgroup_info}
        kickgroups          List all KickGroups
        kickgroup_info      Show the info of a given KickGroup


Function ``kickgroups``:

.. code-block:: none

    usage: kickgroups.py kickgroups [-h] [--root ROOT]
                                    [--sort {TIMESTAMP,KICKGROUP}]

    KickGroups

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           KickGroups Root-Directory
      --sort {TIMESTAMP,KICKGROUP}
                            Sort KickGroups


Function ``kickgroup_info``:

.. code-block:: none

    usage: kickgroups.py kickgroup_info [-h] [--root ROOT] name

    KickGroup Info

    positional arguments:
      name         KickGroup name

    optional arguments:
      -h, --help   show this help message and exit
      --root ROOT  KickGroups Root-Directory

"""
import argparse
import json

from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from dateutil import tz
from pandas import DataFrame
from tfs import TfsDataFrame

# fmt: off
from pylhc.constants.kickgroups import (
    AMPX,
    AMPY,
    AMPZ,
    BEAM,
    BEAMPROCESS,
    BUNCH,
    COLUMNS_TO_HEADERS,
    DRIVEN_TUNEX,
    DRIVEN_TUNEY,
    DRIVEN_TUNEZ,
    KICK_COLUMNS,
    KICK_GROUP_COLUMNS,
    KICKGROUP,
    KICKGROUPS_ROOT,
    LOCALTIME,
    OPTICS,
    OPTICS_URI,
    SDDS,
    TIMESTAMP,
    TUNEX,
    TUNEY,
    TURNS,
    UTCTIME,
)

# fmt: on

# List Kickgroups --------------------------------------------------------------


def kickgroups(by=TIMESTAMP, root: Union[Path, str] = KICKGROUPS_ROOT) -> DataFrame:
    """List all available KickGroups in `root` with optional sorting..

    Args:
        by (str): Column to sort the KickGroups by.
                  Should be either ``TIMESTAMP`` or ``KICKGROUP``.
        root (Path): Alternative path to the KickGroup folder.
                     (Default is set to nfs-path)

    """
    kickgroup_paths = get_all_json_files(root)
    df_info = DataFrame(index=range(len(kickgroup_paths)), columns=KICK_GROUP_COLUMNS)
    for idx, kick_group in enumerate(kickgroup_paths):
        data = load_json(kick_group)
        df_info.loc[idx, KICKGROUP] = data["groupName"]
        df_info.loc[idx, TIMESTAMP] = data["groupCreationTime"]
        df_info.loc[idx, UTCTIME] = ts_to_datetime(df_info.loc[idx, TIMESTAMP])
        df_info.loc[idx, LOCALTIME] = utc_to_local(df_info.loc[idx, UTCTIME])
    df_info = df_info.sort_values(by=by).set_index(TIMESTAMP)
    print(df_info.to_string(index=False, formatters=time_formatters(), justify="center"))
    return df_info


def get_all_json_files(root: Union[Path, str] = KICKGROUPS_ROOT) -> List[Path]:
    """Returns a list of all json files in the folder."""
    return list(Path(root).glob("*.json"))


# Kickgroup Info ---------------------------------------------------------------


def kickgroup_info(kick_group: str, root: Union[Path, str] = KICKGROUPS_ROOT) -> TfsDataFrame:
    """Gather all important info about the KickGroup into a TfsDataFrame and print it.

    Args:
        kick_group (str): KickGroup name.
        root (Path): Alternative path to the KickGroup folder.
                     (Default is set to nfs-path)
    """
    kick_group_data = load_json(Path(root) / f"{kick_group}.json")
    kicks_files = kick_group_data["jsonFiles"]
    df_info = TfsDataFrame(index=range(len(kicks_files)), columns=KICK_COLUMNS, headers={KICKGROUP: kick_group})
    for idx, kf in enumerate(kicks_files):
        df_info.loc[idx, :] = load_kickfile(kf)

    for column in COLUMNS_TO_HEADERS:
        df_info.headers[column] = df_info[column][0]

    _print_kickgroup_info(df_info)
    return df_info


def load_kickfile(kickfile: Union[Path, str]) -> pd.Series:
    """Load the important data from the kickfile-json.

    Args:
        kickfile (Path or str): Path to the kickfile.
    """
    kick = load_json(kickfile)
    data = pd.Series(index=KICK_COLUMNS, dtype=object)
    data[LOCALTIME] = jsontime_to_datetime(kick["acquisitionTime"])
    data[UTCTIME] = local_to_utc(data[LOCALTIME])
    data[SDDS] = kick["sddsFile"]
    data[BEAM] = kick["measurementEnvironment"]["lhcBeam"]["beamName"]
    data[BEAMPROCESS] = kick["measurementEnvironment"]["environmentContext"]["name"]
    data[TURNS] = kick["acqSettings"]["capturedTurns"]
    data[BUNCH] = kick["acqSettings"]["bunchSelection"]
    data[OPTICS] = kick["measurementEnvironment"]["opticsModel"]["opticName"]
    data[OPTICS_URI] = kick["measurementEnvironment"]["opticsModel"]["opticModelURI"]

    three_d = "3D" in kick["excitationSettings"][0]["type"]

    if three_d:
        data[TUNEX] = kick["excitationSettings"][0]["acDipoleSettings"][0]["measuredTune"]
        data[TUNEY] = kick["excitationSettings"][0]["acDipoleSettings"][1]["measuredTune"]
        data[DRIVEN_TUNEX] = data[TUNEX] + kick["excitationSettings"][0]["acDipoleSettings"][0]["deltaTuneStart"]
        data[DRIVEN_TUNEY] = data[TUNEY] + kick["excitationSettings"][0]["acDipoleSettings"][1]["deltaTuneStart"]
        data[DRIVEN_TUNEZ] = kick["excitationData"][0]["rfdata"]["excitationFrequency"]
        data[AMPX] = kick["excitationSettings"][0]["acDipoleSettings"][0]["amplitude"]
        data[AMPY] = kick["excitationSettings"][0]["acDipoleSettings"][1]["amplitude"]
        data[AMPZ] = kick["excitationSettings"][0]["longitudinalRfSettings"]["excitationAmplitude"]
    else:
        data[TUNEX] = kick["excitationSettings"][0]["measuredTune"]
        data[TUNEY] = kick["excitationSettings"][1]["measuredTune"]
        data[DRIVEN_TUNEX] = data[TUNEX] + kick["excitationSettings"][0]["deltaTuneStart"]
        data[DRIVEN_TUNEY] = data[TUNEY] + kick["excitationSettings"][1]["deltaTuneStart"]
        data[DRIVEN_TUNEZ] = np.NaN
        data[AMPX] = kick["excitationSettings"][0]["amplitude"]
        data[AMPY] = kick["excitationSettings"][1]["amplitude"]
        data[AMPZ] = np.NaN

    return data


def _print_kickgroup_info(kicks_info: TfsDataFrame):
    """Actually print the full info about the kickgroup."""
    for header, value in kicks_info.headers.items():
        print(f"{header}: {value}")
    print()
    print(
        kicks_info.drop(columns=COLUMNS_TO_HEADERS).to_string(
            index=False, na_rep=" - ", justify="center", formatters=time_formatters()
        )
    )


# Helper -----------------------------------------------------------------------

# IO ---


def load_json(jsonfile: Union[Path, str]) -> dict:
    return json.loads(Path(jsonfile).read_text())


# Time ---


def ts_to_datetime(ts: int) -> datetime:
    return datetime.utcfromtimestamp(ts / 1000)


def jsontime_to_datetime(time_str: str) -> datetime:
    return datetime.strptime(time_str, "%d-%m-%y_%H-%M-%S")


def datetime_to_string(dt: datetime):
    return dt.strftime("  %Y-%m-%d %H:%M:%S")


def time_formatters():
    return {UTCTIME: datetime_to_string, LOCALTIME: datetime_to_string}


def utc_to_local(dt: datetime):
    return dt.replace(tzinfo=tz.gettz("UTC")).astimezone(tz.gettz("Europe/Paris"))


def local_to_utc(dt: datetime):
    return dt.replace(tzinfo=tz.gettz("Europe/Paris")).astimezone(tz.gettz("UTC"))


# Script Mode ------------------------------------------------------------------


def get_args():
    """Parse Commandline Arguments."""
    # argparse is a bit weird: you need to create the normal parser
    # AND a parent parser if you want to use the same arg in both subparsers.
    # Using the main parser also as parent will result in `function` being
    # always `None`.
    parser = argparse.ArgumentParser(description="KickGroups Functions")
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument(
        "--root", type=str, required=False, default=KICKGROUPS_ROOT, dest="root", help="KickGroups Root-Directory"
    )
    subparsers = parser.add_subparsers(title="Functionality", dest="function")
    parser_kickgroups = subparsers.add_parser(
        "kickgroups", parents=[parent_parser], add_help=False, description="KickGroups", help="List all KickGroups"
    )
    parser_kickgroups.add_argument(
        "--sort",
        type=str,
        dest="sort",
        help="Sort KickGroups",
        choices=[TIMESTAMP, KICKGROUP],
        default=TIMESTAMP,
    )
    parser_info = subparsers.add_parser(
        "kickgroup_info",
        parents=[parent_parser],
        add_help=False,
        description="KickGroup Info",
        help="Show the info of a given KickGroup",
    )
    parser_info.add_argument("name", type=str, help="KickGroup name")
    return parser.parse_args()


if __name__ == "__main__":
    options = get_args()
    if options.function == "kickgroups":
        kickgroups(by=options.sort, root=options.root)

    if options.function == "kickgroup_info":
        kickgroup_info(kick_group=options.name, root=options.root)
