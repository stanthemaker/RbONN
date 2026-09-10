"""The calibration physics: arrays in, fitted parameters out.

Nothing in here imports a driver, and nothing in here should start.  That is
the whole point of the subpackage -- these fits are the part that fails quietly
and wrongly, so they have to stay testable against saved CSVs with the bench
powered down.  Anything that needs to ask an instrument for a number belongs in
``calibration_module.measure``; call in here afterwards with the rows it
collected.

The JSON and CSV readers live beside the fit that defines each format, because
a file layout is part of that fit's contract with the next step.
"""
