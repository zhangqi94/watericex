import sys
sys.path.append("../")
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
from typing import Callable, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from anastrutools import jax_linear_interpolation, load_txt_data
from src.units import units, mass_H_list

####################################################################################################

def get_physical_constants(isotope: int = 0) -> float:
    """
    Calculate physical constants for the given hydrogen isotope.
    """
    mH = mass_H_list[isotope] * units.amu_2_kg
    hbar = units.h_in_J / (2 * np.pi)
    meV_to_J = units.eV_2_J * units.meV_2_eV

    h2_over_2m = (hbar**2) / (2 * mH * units.angstrom_2_m**2 * meV_to_J)
    return h2_over_2m 


####################################################################################################

def make_v_potential(pot_file: str) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """
    Generate the potential energy function from input data.
    """
    data = load_txt_data(pot_file)
    list_d, list_s = data[:, 1], data[:, 2]
    num_d = len(list_d)
    list_s -= list_s[num_d // 2]

    #in the unit of meV
    v_potential = jax.jit(jax_linear_interpolation(jnp.array(list_d), 
                                                   jnp.array(list_s) / units.meV_2_eV)
                          )
    
    print("well depth:", - jnp.min(list_s) / units.meV_2_eV)
    return v_potential, list_d, list_s


def make_v_potential_ref():
    bohr_2_angstrom = 0.52917721067
    hartree_to_meV = 27.21138386 * 1000
    
    def v_potential(x): #in the unit of meV
        x = x / bohr_2_angstrom
        v = (0.1100*x**4 - 0.0475*x**2 + 0.0051) * hartree_to_meV
        return v
        
    list_d = jnp.linspace(-0.45, 0.45, 1001)
    list_s = v_potential(list_d)
    
    return v_potential, list_d, list_s

####################################################################################################

def build_hamiltonian(
    v_potential: Callable,
    x_mesh: np.ndarray,
    n_mesh: int,
    step_size: float,
    h2_over_2m: float
) -> jnp.ndarray:
    """
    Construct the Hamiltonian matrix for a discrete mesh.

    Args:
        v_potential (Callable): Potential energy function.
        x_mesh (np.ndarray): Mesh grid points.
        n_mesh (int): Number of grid points.
        step_size (float): Mesh spacing (h).
        h2_over_2m (float): Physical constant hbar^2 / (2*m).

    Returns:
        jnp.ndarray: Hamiltonian matrix.
    """
    v_diag = jnp.diag(v_potential(x_mesh), 0)
    kinetic_diag = h2_over_2m * 2 * (
        -2 * jnp.diag(jnp.ones(n_mesh, dtype=jnp.float64), 0)
        + jnp.diag(jnp.ones(n_mesh - 1, dtype=jnp.float64), 1)
        + jnp.diag(jnp.ones(n_mesh - 1, dtype=jnp.float64), -1)
    ) / (2 * step_size**2)
    return v_diag - kinetic_diag

####################################################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # File containing potential data
    # potential_file = "data_double_well/a_3.000.txt"
    # potential_file = "data_double_well/a_2.909.txt"
    potential_file = "data_double_well/a_2.800.txt"
    potential_file = "data_double_well/a_2.776.txt"
    # potential_file = "data_double_well/a_2.600.txt"
    
    # potential_file = "data_double_well/a_2.909331.txt"
    # potential_file = "data_double_well/a_3.100000.txt"
    
    n_mesh = 1000
    # num_of_orbs = 5
    num_of_orbs = 2
        
    # Compute physical constants
    h2_over_2m = get_physical_constants(isotope=0)
    print("h2_over_2m:", h2_over_2m)

    # Load potential data
    v_potential, distances, energies = make_v_potential(potential_file)
    # v_potential, distances, energies = make_v_potential_ref()

    # Define the mesh grid
    x_min, x_max = distances[0], distances[-1]
    x_mesh = np.linspace(x_min, x_max, n_mesh)
    step_size = x_mesh[1] - x_mesh[0]

    # Build the Hamiltonian
    hamiltonian = build_hamiltonian(v_potential, x_mesh, n_mesh, step_size, h2_over_2m)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    w = eigenvalues
    v = eigenvectors
    print(f"Lowest {num_of_orbs} Energies:(eV)")
    print(f"{w[:num_of_orbs:]}")

    # --------------------------------------------------------------------------------------------
    # Plot 1: Ground State Potential and Density
    # --------------------------------------------------------------------------------------------
    # Extract ground state wavefunction and normalize
    gs_wf = eigenvectors[:, 0]
    normalization_factor = (gs_wf**2).sum() * step_size
    gs_density = gs_wf**2 / normalization_factor
    
    zoom_fact = 100
    plt.figure(figsize=(6, 4.5), dpi=300)
    plt.title(potential_file)
    plt.grid(1)
    plt.plot(x_mesh, [v_potential(x) for x in x_mesh], '-', color='black', lw=2, label="potential")
    plt.plot(x_mesh, (gs_density) * zoom_fact + w[0], color='grey', label='density', lw=1)
    plt.plot([-0.5, 0.5], [w[0], w[0]], '-', lw=1, label=f"E{0}={w[0]:.2f} meV")
    plt.xlabel(r"position (A)")
    plt.ylabel(r"energy (meV)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    
    # --------------------------------------------------------------------------------------------
    # Plot 2: Excited State Wavefunctions
    # --------------------------------------------------------------------------------------------
    zoom_fact = 50
    plt.figure(figsize=(6, 4.5), dpi=300)
    plt.title(potential_file)
    plt.grid(1)
    plt.plot(x_mesh, [v_potential(x) for x in x_mesh], '-', color='black', lw=2, label="potential")
    for i in range(num_of_orbs):
        # ground state wave function
        gs_wf = v[:,i]
        normalize_factor = (gs_wf**2).sum() * step_size
        # probability density
        exact_phi = gs_wf / jnp.sqrt(normalize_factor)
        plt.plot(x_mesh, exact_phi*zoom_fact + w[i], lw = 1, label=f"E{i}={w[i]:.2f} meV")
    plt.xlabel(r"position (A)")
    plt.ylabel(r"energy (meV)")
    # plt.ylim([-350, 450])
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    

    # --------------------------------------------------------------------------------------------
    # Plot 3: Excited State Wavefunctions
    # --------------------------------------------------------------------------------------------
    zoom_fact = 200
    plt.figure(figsize=(6, 4.5), dpi=300)
    plt.title(potential_file)
    plt.grid(1)
    plt.plot(x_mesh, [v_potential(x) for x in x_mesh], '-', color='black', lw=2, label="potential")
    for i in range(num_of_orbs):
        # ground state wave function
        gs_wf = v[:,i]
        normalize_factor = (gs_wf**2).sum() * step_size
        # probability density
        exact_phi = gs_wf / jnp.sqrt(normalize_factor)
        plt.plot(x_mesh, exact_phi*zoom_fact, lw = 1, label=f"E{i}={w[i]:.2f} meV")
        plt.plot([-0.5, 0.5], [w[i], w[i]], '-', lw=1, label=f"E{i}={w[i]:.2f} meV")
        
    plt.xlabel(r"position (A)")
    plt.ylabel(r"energy (meV)")
    # plt.ylim([-350, 450])
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    
