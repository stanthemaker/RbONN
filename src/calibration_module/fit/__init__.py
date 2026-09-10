"""The calibration physics: arrays in, fitted parameters out.

Nothing in here imports a driver, and nothing in here should start.  That is
the whole point of the subpackage -- these fits are the part that fails quietly
and wrongly, so they have to stay testable against saved CSVs with the bench
powered down.  Anything that needs to ask an instrument for a number belongs in
``calibration_module.measure``; call in here afterwards with the rows it
collected.

    pair_v2   step 6 -- per-pair eta from the difference estimator, and the
              PairV2Config that says what grid it was specified with
    pair      step 6 v1 -- the joint 6-parameter grid fit pair_v2 replaced.
              Reached only by calib_step6_v1.py and calib_synth_v1.py now;
              kept so historical CSVs can be re-fit both ways
    phase     steps 7 + 8 -- comb phase dPhi_comb, and the forward model
    center    centre-wavelength scan -- weighted quadratic vertex fit
    sigma     the shared systematic std floor, imported by all three v2 steps
    report    fringe / residual plots over the fits above

The JSON and CSV readers live beside the fit that defines each format, because
a file layout is part of that fit's contract with the next step.

Plot helpers here take a Matplotlib ``Figure`` the caller owns (``make_plot``,
``plot_fringe``) rather than making one, so a GUI canvas and a headless PNG
render from one implementation; ``save_plot`` is the PNG wrapper.
"""
