import os
import numpy as np

from simulations.soss_simulations import run_simulations_random

output_directory = "/data/typhon2/hattie/jwst/soss_simulations/"
number_samples = 10

simulated_spectra = run_simulations_random(
    number_samples,
    os.path.join(output_directory, f"{number_samples}_soss_sims_randomised_target.h5"),
    N_contaminants="random",
    )
