"""Calibration physics: the models, fits and file formats behind the calib steps.

Instrument-free by design.  Nothing in the pure modules below imports the SLM,
the DAQ or the OSA -- they take arrays of commanded levels and measured volts,
and return fitted parameters.  A calibration *script* is what dispatches work
to ``slm_module`` (display these (x, w) tuples) and ``daq_module`` (read a
point), collects the rows, and then calls in here to fit them.

    pair      step 6 -- per-pair TPA efficiency eta, from the reduced x/w curves
    phase     steps 7 + 8 -- comb phase dPhi_comb, and the forward model built
              from the step-6 pair fits
    center    centre-wavelength scan -- weighted quadratic vertex fit
    report    fringe / residual plots shared by the phase steps

The ``measure_*`` modules alongside them are LEGACY sweep drivers that still
talk to the SLM and monitor, kept only so the current GUI pages and pipeline
stages keep running.  Do not import them from new code.
"""
