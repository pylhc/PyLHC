"""
Constants: Machine Settings Info
---------------------------------

Constants for the machine settings info.

:module: constants.machine_settings_info

"""
from pylhc.constants.general import TFS_SUFFIX
# TFS-File Conventions #########################################################
# Filename
info_name = f"machine_settings{TFS_SUFFIX}"
knobdef_suffix = f"_definition{TFS_SUFFIX}"

# Columns
column_knob = "KNOB"
column_value = "VALUE"

# Headers
head_time = "Time"
head_beamprocess = "Beamprocess"
head_fill = "Fill"
head_beamprocess_start = "BP_Start"
head_context_category = "ContextCategory"
head_beamprcess_description = "Description"
head_optics = "Optics"
head_optics_start = "Optics_Start"

