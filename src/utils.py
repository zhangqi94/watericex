import numpy as np

####################################################################################################


def get_lorentz_hist(energy_levels, 
                     lx=np.linspace(0, 1000, 10000), 
                     gamma=5.0,
                     ):

    def lorentz(E, E0, gamma):
        return (1 / np.pi) * (gamma / ((E - E0)**2 + gamma**2))
    
    dim = 3
    num_modes = energy_levels.shape[0]
    num_atoms = num_modes // dim + 1
    ly = np.zeros_like(lx)
    for i in range(len(energy_levels)):
        ly += lorentz(lx, energy_levels[i], gamma)
    ly = ly / num_atoms
    
    return lx, ly


