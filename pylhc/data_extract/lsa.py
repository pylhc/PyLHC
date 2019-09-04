"""
PyLSA
---------------------------

Provides additional functionality for pjlsa.


:module: data_extract.lsa
:author: jdilly

"""
import logging
import re
LOG = logging.getLogger(__name__)

import pjlsa

from utils.time_tools import AccDatetime

LOG = logging.getLogger(__name__)

RELEVANT_BP_CONTEXTS = ("OPERATIONAL", "MD")
RELEVANT_BP_CATEGORIES = ("DISCRETE",)


class LSAClient(pjlsa.LSAClient):
    """ Extension of the LSAClient. """

    def find_knob_names(self, accelerator:str = 'lhc', regexp: str = '') -> list:
        """ Return knobs for accelerator.

        Args:
            accelerator: Accelerator name
            regexp: Rgular Expression to filter knob names

        Returns:
            Sorted list of knob names.

        """
        req = pjlsa.ParametersRequestBuilder()
        req.setAccelerator(pjlsa.Accelerators.get(accelerator, accelerator))
        req.setParameterTypeName('KNOB')
        lst = self._parameterService.findParameters(req.build())
        reg = re.compile(regexp, re.IGNORECASE)
        return sorted(filter(reg.search, [pp.getName() for pp in lst]))

    def find_last_fill(self, acc_time: AccDatetime, accelerator: str = 'lhc') -> (str, list):
        """ Return last fill name and content.

         Args:
            acc_time: (AccDatetime): Accelerator datetime object
            accelerator (str): Name of the accelerator

        Returns:
            tupel: Last fill name (str), Beamprocesses of last fill (list)

         """
        start_time = acc_time.sub(days=1)  # assumes a fill is not longer than a day
        try:
            fills = self.findBeamProcessHistory(t1=start_time.local_string(),
                                                t2=acc_time.local_string(),
                                                accelerator=accelerator)
        except TypeError:
            raise ValueError(f"No beamprocesses found in the day before {acc_time.cern_utc_string()}")
        last_fill = sorted(fills.keys())[-1]
        return last_fill, fills[last_fill]

    def find_trims_at_time(self, beamprocess: str, knobs: list, acc_time: AccDatetime, accelerator: str = 'lhc') -> dict:
        """ Get trims for knobs at specific time.

        Args:
            beamprocess (str): Name of the beamprocess
            knobs (list): List of strings of the knobs to check
            acc_time: (AccDatetime): Accelerator datetime object
            accelerator (str): Name of the accelerator

        Returns:
            Dictionary of knob names and their values

        """
        if knobs is None or len(knobs) == 0:
            knobs = self.find_knob_names(accelerator)
        trims = self.getTrims(beamprocess, knobs, end=acc_time.timestamp())
        trims_not_found = [k for k in knobs if k not in trims.keys()]
        if len(trims_not_found):
            LOG.warn(f"The following knobs were not found in '{beamprocess}': {trims_not_found}")
        return {trim: trims[trim].data[-1] for trim in trims.keys()}  # return last set value

    def get_beamprocess_info(self, beamprocess: str):
        """ Get context info of the given beamprocess.

        Args:
            beamprocess (str): Name of the beamprocess.

        Returns:
            Dictionary with context info.
        """
        bp = self._contextService.findStandAloneBeamProcess(beamprocess)
        return _beamprocess_to_dict(bp)

    def find_active_beamprocess_at_time(self, acc_time: AccDatetime, accelerator: str = 'lhc') -> str:
        """ Find the active beam process at the time given.

        Same as what online model extractor (KnobExtractor) does, but returns empty map for some reason.
        """
        raise NotImplementedError("This function does not work yet!")

        if accelerator != 'lhc':
            raise NotImplementedError("Active-Beamprocess retrieval is only implemented for LHC")
        beamprocessmap = self._lhcService.findResidentStandAloneBeamProcessesByTime(int(acc_time.timestamp()))
        # print(str(beamprocessmap))
        beamprocess = beamprocessmap.get("POWERCONVERTERS")
        LOG.debug(f"Active Beamprocess at time '{acc_time.cern_utc_string()}': {beamprocess}")
        return beamprocess


# Single Instance LSAClient ####################################################


class LSAMeta(type):
    """ Metaclass for single instance LSAClient. """
    _client = None

    def __getattr__(cls, attr):
        if cls._client is None:
            LOG.debug("Creating LSA Client (only once).")
            cls._client = LSAClient()

        client_attr = cls._client.__getattribute__(attr)
        if callable(client_attr):
            def hooked(*args, **kwargs):
                result = client_attr(*args, **kwargs)
                if result == cls._client:
                    # prevent client from becoming unwrapped
                    return cls
                return result
            return hooked
        else:
            return client_attr


class LSA(metaclass=LSAMeta):
    """ Import this class to use LSA like the client without the need to instantiate it."""
    pass


# Helper Functions #############################################################


def _beamprocess_to_dict(bp):
    """ Converts some fields of the beamprocess (java) to a dictionary """
    variables = ["category", "contextCategory", "description", "contextFamily", "user"]
    return {var: str(bp.__getattribute__(var)) for var in variables}
