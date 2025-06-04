import sys
sys.path.append("../")
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import ase
import ase.io
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.coordtrans import make_mace_calculator
from main_gen_ice_structure import create_ice
from anastrutools import jax_linear_interpolation

####################################################################################################
# Calculate energy for different `a` and `d`
def main_various_lattice_constants(calc):
    list_a = np.arange(2.4, 3.0, 0.10)
    list_a = np.array([5.710456, 5.434057, 5.248239])/2
    list_d = np.linspace(-0.4, 0.4, 101)
    num_n = 16
    
    num_a, num_d = len(list_a), len(list_d)
    list_e = np.zeros((num_a, num_d))

    for i, a in enumerate(list_a):
        for j, d in enumerate(list_d):
            atoms = create_ice(a=a, d=d)
            atoms.calc = calc
            energy = atoms.get_potential_energy() / (num_n * 2)
            list_e[i, j] = energy
            print(f"a = {a:.3f}, d = {d:.3f}, energy = {energy:.6f}")

    # Plot energy variations
    plt.figure(figsize=(6, 4), dpi=300)
    plt.grid(True)
    for i, a in enumerate(list_a):
        list_e_i = list_e[i] - list_e[i, num_d // 2]
        plt.plot(list_d, list_e_i, "-", linewidth=1.0, label=f"a={a:.2f}",)
    plt.xlabel(r"$d$ / \AA")
    plt.ylabel(r"$e$ / eV")
    plt.ylim([-0.3, 1])
    plt.legend(fontsize=8)
    plt.show()


####################################################################################################

def main_fit_potential(calc, fit="jax"):

    # a, list_d = 3.00, np.linspace(-0.50, 0.50, 401)
    # a, list_d = 2.95, np.linspace(-0.48, 0.48, 401)
    # a, list_d = 2.90, np.linspace(-0.45, 0.45, 401)
    # a, list_d = 2.85, np.linspace(-0.45, 0.45, 401)
    # a, list_d = 2.80, np.linspace(-0.40, 0.40, 401)
    # a, list_d = 2.75, np.linspace(-0.38, 0.38, 401)
    # a, list_d = 2.70, np.linspace(-0.35, 0.35, 401)
    # a, list_d = 2.65, np.linspace(-0.35, 0.35, 401)
    # a, list_d = 2.60, np.linspace(-0.35, 0.35, 401)
    
    a, list_d = 6.2/2, np.linspace(-0.6, 0.6, 401) # 30GPa
    # a, list_d = 5.818662/2, np.linspace(-0.5, 0.5, 101) # 30GPa
    # a, list_d = 5.710456/2, np.linspace(-0.45, 0.45, 101) # 40GPa
    # a, list_d = 5.489954/2, np.linspace(-0.40, 0.40, 101) # 70GPa
    # a, list_d = 5.434057/2, np.linspace(-0.40, 0.40, 101) # 80GPa
    # a, list_d = 5.248239/2, np.linspace(-0.40, 0.40, 101) # 120GPa

    #### calculate energy for different d
    num_d = len(list_d)
    num_n = 16
    list_e = np.zeros((num_d, ))
    
    f = open(f"data_double_well/a_{a:.6f}.txt", "w")
    for j in range(num_d):
        d = list_d[j]
        atoms = create_ice(a = a, d = d)
        atoms.calc = calc
        e = atoms.get_potential_energy() / (num_n*2)
        list_e[j] = e
        print("a = %.3f, d = %.3f, energy = %.6f" % (a, d, e))
        f.write("%.3f  %.6f  %.12f\n" %(a, d, e))
        
    f.close()
    
    # Calculate energy shift for fitting
    list_s = list_e - list_e[num_d//2]
    if fit == "poly":
        #### fit potential: polynomial
        degree = 100
        coeffs = np.polyfit(list_d, list_s, degree )
        fn_pot = np.poly1d(coeffs)
        fit_d = np.linspace(min(list_d), max(list_d), 1001)
        fit_s = fn_pot(fit_d)
        print("fit coefficients:", coeffs)
        
    elif fit == "spline":
        #### fit potential: spline
        from scipy.interpolate import interp1d
        fn_pot = interp1d(list_d, list_s, kind='linear', fill_value="extrapolate")
        fit_d = np.linspace(min(list_d), max(list_d), 1001)
        fit_s = fn_pot(fit_d)

    elif fit == "jax":
        #### fit potential: jax
        fn_pot = jax.jit(jax_linear_interpolation(jnp.array(list_d), jnp.array(list_s)))
        fit_d = jnp.linspace(min(list_d), max(list_d), 1001)
        fit_s = fn_pot(fit_d)
        

    ###### plot figure
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4), dpi=300)
    plt.grid(True)
    plt.plot(fit_d, fit_s, "-", linewidth=1.0, label=f"fit", color="red")
    plt.scatter(list_d, list_s, marker = "+", s = 10, label=f"a={a:.2f}")
    plt.xlabel(r"$d$ / \AA")
    plt.ylabel(r"$e$ / eV")
    # plt.xlim([-0.45, 0.45])
    # plt.ylim([-5, 20])
    # plt.ylim([-10, 20])
    plt.legend(fontsize=8)
    plt.show()




####################################################################################################
if __name__ == '__main__':
    # Initialize the MACE calculator
    mace_model_path = "/home/zq/zqcodeml/waterice/src/macemodel/mace_iceX_l1x128r4.0.model"
    calc = make_mace_calculator(mace_model_path)
    
    # main_various_lattice_constants(calc)
    # main_fit_potential(calc, fit="poly")
    main_fit_potential(calc, fit="jax")






