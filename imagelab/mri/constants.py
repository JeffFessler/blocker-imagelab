# import numpy as np
from numpy import pi


class T2(object):
    # Nishimura [BHAAM87]
    gray_matter = 100  # ms
    white_matter = 92
    muscle = 47
    fat = 85
    kidney = 58
    liver = 43


# Gyromagnetic Ratio (for 1H)
gamma_bar = 42.575  # Mhz/Tesla
gamma = 2 * pi * gamma_bar  # radians/Tesla

# Planck's Constant
planck = 6.626e-34  # J s
planck_bar = planck / (2 * pi)

boltzman = 1.381e-23  # J/K
mu0 = 400 * pi  # nH/m
