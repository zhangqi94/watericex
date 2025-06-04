import sys
sys.path.append("../")

import numpy as np
from mace.calculators import MACECalculator
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.units import units
from src.crystal import create_ice_crystal
from src.coordtrans import make_mace_calculator, calculate_hessian_matrix, calculate_Dmat_eigs
from src.utils import get_lorentz_hist
import matplotlib.pyplot as plt

####################################################################################################
def calculate_phonon_freqs(calc, 
                           init_stru_path, 
                           get_image=False,
                           lx = np.linspace(0, 4000, 10000),
                           ):
    isotope = "H2O"
    dim = 3
    gamma = 20.0
    
    atoms, box_lengths, positions_init, num_molecules, density \
                    = create_ice_crystal(init_stru_path, isotope = isotope,)
    
    Dmat, mass_vec = calculate_hessian_matrix(atoms, calc)
    Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes = calculate_Dmat_eigs(Dmat, mass_vec)
    
    wfreqs_invcm = wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm
    
    if get_image:
        wfreqs = np.sign(wsquares[dim:]) * np.sqrt(np.abs(wsquares[dim:]))
        wfreqs_invcm = wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm

    lx, ly = get_lorentz_hist(wfreqs_invcm, lx=lx, gamma=gamma)
    
    return lx, ly

####################################################################################################
if __name__ == '__main__':
    key = jax.random.key(42)
    isotope = "H2O"
    dim = 3
    
    # mace_model_path = "/home/zq/zqcodeml/waterice/src/macemodel/mace_iceX_l1x128r4.0.model"
    mace_model_path = "/home/zq/zqcodeml/waterice/src/macemodel/mace_iceX_l1x64r4.0.model"
    calc = make_mace_calculator(mace_model_path)
    
    ## ice08
    # init_stru_path = "/home/zq/zqcodeml/waterice/src/structures/relax_l1x128r4.0/cubic_ice08_n016_p60.00.vasp"
    
    # lx, ly = calculate_phonon_freqs(calc, init_stru_path, get_image=True)
    
    # plt.figure(figsize=(6, 4), dpi=300)
    # plt.plot(lx, ly,  "-", linewidth=1.0, label="energy")
    
    # plt.title(init_stru_path)
    # plt.grid(True)
    # plt.xlabel(r"frequencies (cm$^{-1}$)")
    # plt.ylabel(r"density of states")
    # plt.show()

    list_p = np.arange(30, 100, 10)    
    # list_p = [200]

    for p in list_p:
        
        print(f"p={p:.2f}")

        # init_stru_path = f"/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n016_p{p:.2f}.vasp"
        init_stru_path = f"/home/zq/zqcodeml/waterice/src/structures/relax/ice08c_n128_p{p:.2f}.vasp"
        
        lx, ly = calculate_phonon_freqs(calc, init_stru_path, get_image=True)
    

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(lx, ly,  "-", linewidth=1.0, label=f"p={p:.2f}")
        plt.xlabel(r"frequencies (cm$^{-1}$)")
        plt.ylabel(r"density of states")
        plt.legend()
        plt.grid(1)
        plt.show()
