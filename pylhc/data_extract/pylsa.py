"""
PyLSA
---------------------------

Provides additional functionality for pjlsa.

:module: data_extract.pylsa
:author: jdilly

"""
import pjlsa
import re
import logging

LOG = logging.getLogger(__name__)


class LSAClient(pjlsa.LSAClient):
    """ Extension of the LSAClient. """

    def find_knob_names(self, accelerator='lhc', regexp=''):
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

    def get_last_fill(self, acc_time, accelerator='lhc'):
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
            raise ValueError(f"No beamprocesses found in the day before {acc_time.utc_string()}")
        last_fill = sorted(fills.keys())[-1]
        return last_fill, fills[last_fill]

    def get_trims_at_time(self, beamprocess, knobs, acc_time, accelerator='lhc'):
        """ Get trims for knobs at specific time.

        Args:
            beamprocess (str): Name of the beamprocess
            knobs (list): List of strings of the knobs to check
            acc_time: (AccDatetime): Accelerator datetime object
            accelerator (str): Name of the accelerator

        Returns:
            Dictionary of knob names and their values

        """
        if len(knobs) == 0:
            knobs = self.find_knob_names(accelerator)
        trims = self.getTrims(beamprocess, knobs, end=acc_time.timestamp())
        trims_not_found = [k for k in knobs if k not in trims.keys()]
        if len(trims_not_found):
            LOG.warn(f"The following knobs were not found in '{beamprocess}': {trims_not_found}")
        return {trim: trims[trim].data[-1] for trim in trims.keys()}  # return last set value
