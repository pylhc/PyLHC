IR_POS_DEFAULT = {
    "LHCB1": {
        'IP1': 23519.36962,
        'IP2': 192.923,
        'IP3': 3525.207216,
        'IP4': 6857.491433,
        'IP5': 10189.77565,
        'IP6': 13522.21223,
        'IP7': 16854.64882,
        'IP8': 20175.8654,
    },
    "LHCB2": {
        'IP1': 3195.252584,
        'IP2': 6527.5368,
        'IP3': 9859.973384,
        'IP4': 13192.40997,
        'IP5': 16524.84655,
        'IP6': 19857.13077,
        'IP7': 23189.41498,
        'IP8': 26510.4792,
    }
}
MANUAL_STYLE = {
    # differences to the standard style
    u'lines.markersize': 5.0,
    u'lines.linestyle': u'',
    # u'figure.figsize': [6, 3.8],
}

ERROR_ALPHA = 1.  # Set errorbar transparency
MAX_LEGENDLENGTH = 78  # maximum length of legend letters before linebreak
COMPLEX_NAMES = [p+ext for p in ["1001", "1010"] for ext in "RI"]  # Endings of columns that contain complex data