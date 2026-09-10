"""Acquisition: drive the SLM, the DAQ and the OSA, return the rows.

Thin on purpose.  A sweep here decides what to display and when to read, and
hands back arrays -- it does not fit them; ``calibration_module.fit`` does that,
and keeping the two apart is what lets the fits be tested without hardware.

    bench     connect/read primitives shared by the step scripts

``center``, ``pair`` and ``phase`` are LEGACY sweep drivers with one GUI page
each.  They predate the v2 step scripts and are scheduled to die with the GUI
v2 rebuild; do not import them from new code.  Their replacements land here,
under the same names.
"""
