"""
PyLSA
-----

This module provides useful functions to conveniently wrap the functionality of ``pjlsa``.
"""
from typing import Callable, Union, Dict

import logging
import re
from contextlib import suppress

import tfs
import jpype
from jpype import java, JException
from omc3.utils.mock import cern_network_import, CERNNetworkMockPackage
from omc3.utils.time_tools import AccDatetime

LOG = logging.getLogger(__name__)
pytimber = cern_network_import("pytimber")
pjlsa = cern_network_import("pjlsa")
try:
    pjLSAClient = pjlsa.LSAClient
except ImportError:
    pjLSAClient = object

RELEVANT_BP_CONTEXTS = ("OPERATIONAL", "MD")
RELEVANT_BP_CATEGORIES = ("DISCRETE",)

HEAD_KNOB = "Knob"
HEAD_OPTICS = "Optics"
HEAD_INFO = "Info"
COL_NAME = "NAME"
COL_CIRCUIT = "CIRCUIT"
PREF_DELTA = "DELTA_"

MAX_RETRIES = 10


class LSAClient(pjLSAClient):
    """Extension of the LSAClient."""

    def __getattr__(self, item):
        """ Overwrite __getattr__ to raise the proper import errors at the proper time."""
        try:
            super().__getattr__(item)
        except AttributeError as e:
            pjlsa.pjLSAClient  # might raise the Mock-Class import error
            raise e  # if that worked, raise the actual attribute error

    def find_knob_names(self, accelerator: str = "lhc", regexp: str = "") -> list:
        """
        Return knobs for accelerator.

        Args:
            accelerator: Accelerator name.
            regexp: Regular Expression to filter knob names.

        Returns:
            Sorted list of knob names.
        """
        req = pjlsa.ParametersRequestBuilder()
        req.setAccelerator(pjlsa.Accelerators.get(accelerator, accelerator))
        req.setParameterTypeName("KNOB")
        lst = self._parameterService.findParameters(req.build())
        reg = re.compile(regexp, re.IGNORECASE)
        return sorted(filter(reg.search, [pp.getName() for pp in lst]))

    def find_last_fill(self, acc_time: AccDatetime, accelerator: str = "lhc", source: str = "nxcals") -> (str, list):
        """
        Return last fill name and BeamProcesses.

         Args:
            acc_time: (AccDatetime): Accelerator datetime object.
            accelerator (str): Name of the accelerator.

        Returns:
            tupel: Last fill name (str), Beamprocesses of last fill (list).
         """
        start_time = acc_time.sub(days=1)  # assumes a fill is not longer than a day
        try:
            fills = self.find_beamprocess_history(
                t_start=start_time, t_end=acc_time,
                accelerator=accelerator,
                source=source,
            )
        except TypeError as e:
            raise ValueError(
                f"No beamprocesses found in the day before {acc_time.cern_utc_string()}"
            ) from e
        last_fill = sorted(fills.keys())[-1]
        return last_fill, fills[last_fill]

    def find_beamprocess_history(self, t_start: AccDatetime, t_end: AccDatetime, accelerator="lhc") -> Dict:
        """
        Finds the BeamProcesses between t_start and t_end and sorts then by fills.
        Adapted from pjlsa's FindBeamProcessHistory but with source pass-through
        and trial-loop.

        Args:
            t_start (AccDatetime): start time
            t_end (AccDatetime): end time
            accelerator (str): Name of the accelerator.

        Returns:
            Dictionary of fills (keys) with a list of Timestamps and BeamProcesses.

        """
        cts = self.findUserContextMappingHistory(t_start.timestamp(), t_end.timestamp(), accelerator=accelerator)

        db = pytimber.LoggingDB(source=source)
        fillnts, fillnv = try_to_acquire_data(
            db.get, "HX:FILLN", t_start.timestamp(), t_end.timestamp()
        )["HX:FILLN"]

        fills = {}
        for ts, name in zip(cts.timestamp, cts.name):
            idx = fillnts.searchsorted(ts) - 1
            filln = int(fillnv[idx])
            fills.setdefault(filln, []).insert(0, (ts, name))
        return fills

    def find_trims_at_time(
            self, beamprocess: str, knobs: list, acc_time: AccDatetime, accelerator: str = "lhc"
    ) -> dict:
        """
        Get trims for knobs at specific time.

        Args:
            beamprocess (str): Name of the beamprocess.
            knobs (list): List of strings of the knobs to check.
            acc_time: (AccDatetime): Accelerator datetime object.
            accelerator (str): Name of the accelerator.

        Returns:
            Dictionary of knob names and their values.
        """
        if knobs is None or len(knobs) == 0:
            knobs = self.find_knob_names(accelerator)
        trims = self.getTrims(parameter=knobs, beamprocess=beamprocess, end=acc_time.timestamp())
        trims_not_found = [k for k in knobs if k not in trims.keys()]
        if len(trims_not_found):
            LOG.warning(f"The following knobs were not found in '{beamprocess}': {trims_not_found}")
        trim_dict = {trim: trims[trim].data[-1] for trim in trims.keys()}  # return last set value
        for trim, value in trim_dict.items():
            try:
                trim_dict[trim] = value.flatten()[-1]  # the very last entry ...
            except AttributeError:
                continue  # single value, as expected
            else:
                LOG.warning(f"Trim {trim} hat multiple data entries {value}, taking only the last one.")
        return trim_dict

    def get_beamprocess_info(self, beamprocess: Union[str, object]) -> Dict:
        """
        Get context info of the given beamprocess.

        Args:
            beamprocess (str): Name of the beamprocess.

        Returns:
            Dictionary with context info.
        """
        if isinstance(beamprocess, str):
            beamprocess = self._contextService.findStandAloneBeamProcess(beamprocess)
        bp_dict = _beamprocess_to_dict(beamprocess)
        LOG.debug(str(bp_dict))
        return bp_dict

    def find_active_beamprocess_at_time(
            self, acc_time: AccDatetime, accelerator: str = "lhc",
            bp_group: str = "POWERCONVERTERS"  # the Beamprocesses relevant for OMC, others: 'ADT', 'KICKERS', 'SPOOLS', 'COLLIMATORS'
    ):
        """
        Find the active beam process at the time given.
        Same as what online model extractor (KnobExtractor) does.

        Returns:
            'cern.lsa.domain.settings.spi.StandAloneBeamProcessImpl'
        """
        if accelerator != "lhc":
            raise NotImplementedError("Active-Beamprocess retrieval is only implemented for LHC")

        beamprocessmap = self._lhcService.findResidentStandAloneBeamProcessesByTime(
            int(acc_time.timestamp() * 1000)  # java timestamps are in milliseconds
        )
        beamprocess = beamprocessmap.get(bp_group)
        if beamprocess is None:
            raise ValueError(f"No active BeamProcess found for group '{bp_group}' "
                             f"at time {acc_time.utc_string} UTC.")
        LOG.debug(f"Active Beamprocess at time '{acc_time.cern_utc_string()}': {str(beamprocess)}")
        return beamprocess

    def get_knob_circuits(self, knob_name: str, optics: str) -> tfs.TfsDataFrame:
        """
        Get a dataframe of the structure of the knob. Similar to online model extractor
        (KnobExtractor.getKnobHiercarchy)
        
        Args:
            knob_name: name of the knob.
            optics: name of the optics.

        Returns:
            A `TfsDataFrame` of the knob circuits.
        """
        LOG.debug(f"Getting knob defintions for '{knob_name}', optics '{optics}'")
        df = tfs.TfsDataFrame()
        df.headers[HEAD_KNOB] = knob_name
        df.headers[HEAD_OPTICS] = optics
        df.headers[HEAD_INFO] = "In MAD-X it should be 'name = name + DELTA * knobValue'"
        knob = self._knobService.findKnob(knob_name)
        if knob is None:
            raise IOError(f"Knob '{knob_name}' does not exist")
        try:
            knob_settings = knob.getKnobFactors().getFactorsForOptic(optics)
        except jpype.JException(jpype.java.lang.IllegalArgumentException):
            raise IOError(f"Knob '{knob_name}' not available for optics '{optics}'")

        for knob_factor in knob_settings:
            circuit = knob_factor.componentName
            param = self._parameterService.findParameterByName(circuit)
            type_ = param.getParameterType().getName()
            madx_name = self.get_madx_name_from_circuit(circuit)
            if madx_name is None:
                LOG.error(
                    f"Circuit '{circuit}' could not be resolved to a MADX name in LSA! "
                    "It will not be found in knob-definition!"
                )
            else:
                LOG.debug(f"  Found component '{circuit}': {madx_name}, {knob_factor.factor}")
                df.loc[madx_name, COL_CIRCUIT] = circuit
                df.loc[madx_name, f"{PREF_DELTA}{type_.upper()}"] = knob_factor.factor
        return df.fillna(0)

    def get_madx_name_from_circuit(self, circuit: str):
        """Returns the ``MAD-X`` Strength Name (Circuit/Knob) from the given circuit name."""
        logical_name = circuit.split("/")[0]
        slist = jpype.java.util.Collections.singletonList(
            logical_name
        )  # python lists did not work (jdilly)
        madx_name = self._deviceService.findMadStrengthNamesByLogicalNames(slist)  # returns a map
        return madx_name[logical_name]


# Single Instance LSAClient ####################################################


class LSAMeta(type):
    """Metaclass for single instance LSAClient."""

    _client = None

    def __getattr__(cls, attr):
        if cls._client is None:
            LOG.debug("Creating LSA Client (only once).")
            cls._client = LSAClient()

        client_attr = cls._client.__getattribute__(attr)
        if callable(client_attr):

            def hooked(*args, **kwargs):
                result = client_attr(*args, **kwargs)
                result_is_self = False
                try:
                    if result == cls._client:
                        # prevent client from becoming unwrapped
                        return cls
                except (ValueError, SystemError):
                    # happens with e.g. numpy arrays (ValueError)
                    # or JavaObjects (SytemError) as return values
                    pass
                return result

            return hooked
        else:
            return client_attr


class LSA(metaclass=LSAMeta):
    """Import this class to use LSA like the client without the need to instantiate it.
    Disadvantage: It will always use the default Server.
    """
    pass


# Helper Functions #############################################################


def _beamprocess_to_dict(bp):
    """Converts some fields of the beamprocess (java) to a dictionary."""
    bp_dict = {'Name': bp.toString(), "Object": bp}
    bp_dict.update({getter[3:]: str(bp.__getattribute__(getter)())  # __getattr__ does not exist
                    for getter in dir(bp)
                    if getter.startswith('get') and "Attribute" not in getter})
    return bp_dict


def try_to_acquire_data(function: Callable, *args, **kwargs):
    """Tries to get data from function multiple times.

     Args:
         function (Callable): function to be called, e.g. db.get
         args, kwargs: arguments passed to ``function``

    Returns:
        Return arguments of ``function``

     """
    retries = MAX_RETRIES
    for tries in range(retries + 1):
        try:
            return function(*args, **kwargs)
        except java.lang.IllegalStateException as e:
            raise IOError("Could not acquire data, user probably has no access to NXCALS") from e
        except JException as e:  # Might be a case for retries
            if "RetryableException" in str(e) and (tries + 1) < retries:
                LOG.warning(f"Could not acquire data! Trial no {tries + 1} / {retries}")
                continue  # will go to the next iteratoin of the loop, so retry
            raise IOError("Could not acquire data!") from e
