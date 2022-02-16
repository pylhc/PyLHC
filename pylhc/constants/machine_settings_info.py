"""
Constants: Machine Settings Info
---------------------------------

Specific constants relating to the retrieval of machine settings information to be used in
``PyLHC``, to help with consistency.
"""
from pylhc.constants.general import TFS_SUFFIX

# TFS-File Conventions #########################################################
# Filename
info_name = f"machine_settings{TFS_SUFFIX}"
knobdef_suffix = f"_definition{TFS_SUFFIX}"
trimhistory_suffix = f"_trims{TFS_SUFFIX}"

# Columns
column_knob = "KNOB"
column_time = "TIME"
column_timestamp = "TIMESTAMP"
column_value = "VALUE"


# Headers
head_accel = "Accelerator"
head_time = "Time"
head_start_time = "StartTime"
head_end_time = "EndTime"
head_beamprocess = "Beamprocess"
head_fill = "Fill"
head_beamprocess_start = "BeamprocessStart"
head_context_category = "ContextCategory"
head_beamprcess_description = "Description"
head_optics = "Optics"
head_optics_start = "OpticsStart"
