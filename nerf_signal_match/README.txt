Neural-field signal matching on one ROGII train well.
Put typewell.csv -> tw.csv and horizontal_well.csv -> hw.csv next to the scripts.
Order: python3 base.py (baselines) ; python3 field.py (train 1D scale-conditioned Fourier-feature field on type log) ;
python3 infer.py (calibrate on TVT_input prefix, optimise piecewise-linear structure slope on lateral, coarse-to-fine) ;
python3 diag.py (lambda sweep, placebo controls, plot) ; python3 infer2.py (fault/jump experiment).
NOTE: tops columns (ANCC..BUDA) are NOT used: in this file TVT = 11855.2 + ANCC - Z exactly (leak).
