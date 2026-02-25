# Common Circuit ASC Fixtures

These `.asc` files are baseline fixtures for manual visual cleanup and
subsequent schematic-generation calibration.

Edit these files directly in LTspice to match your preferred layout/style.

Current fixture set:

- `rc_lowpass_ac.asc`
- `rc_highpass_ac.asc`
- `rl_highpass_ac.asc`
- `rlc_series_lowpass_ac.asc`
- `resistor_divider_spec.asc`
- `diode_half_wave_rectifier.asc`
- `bridge_rectifier_filter.asc`
- `zener_regulator_dc.asc`
- `inverting_opamp_ac.asc`
- `non_inverting_opamp_spec.asc`
- `common_emitter_npn.asc`
- `common_source_nmos.asc`

Notes:

- Keep filenames stable so tests/calibration scripts can map each topology.
- Keep circuits functionally equivalent; visual/layout refinements are expected.
- If you want additional topologies, add new `.asc` files in this folder.
- Canonical visual style in this repo:
  - grid-aligned coordinates (`16` units),
  - left-to-right signal flow anchored near `x=120, y=156`,
  - simulation directives grouped at the lower-left (`x=48`) with `24` unit line spacing.
