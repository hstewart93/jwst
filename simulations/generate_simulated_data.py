import os
import numpy as np

from simulations.soss_simulations import run_simulations

output_directory = "/data/typhon2/hattie/jwst/soss_simulations"
number_samples = 10000

run_simulations(
    N_simulations=number_samples,
    targ_Jmag=(0, 10),
    targ_Teff=(2000, 7000),
    N_contaminants=(0, 10),
    output_file=os.path.join(output_directory, f"{number_samples}_soss_sims_randomised_target.h5"),
)
