import numpy as np
from mace.calculators import MACECalculator
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

## import units and constants
try:
    from src.units import units
    from src.potentialmace import make_mace_calculator
except:
    from units import units
    from potentialmace import make_mace_calculator

####################################################################################################
def calculate_hessian_matrix(atoms,
                             calc,
                             ):
    """
    Calculate the Hessian matrix for a system of molecules.
    The calculation involves getting the Hessian matrix, ensuring it is Hermitian, 
    and applying mass weighting to convert it to a form suitable for frequency analysis.

    Args:
    --------
    atoms : object (ase.Atoms)
        The atoms object containing the molecular structure and positions.
    calc : object (ase.calc)
        The calculator object used to compute the Hessian matrix.

    Returns:
    --------
    Cmat : ndarray, shape=(3N, 3N )
        The original Hessian matrix, in the units of eV/A^2.
    Dmat : ndarray, shape=(3N, 3N )
        The mass-weighted Hessian matrix, in the units of eV/(A^2*amu).
    """
    
    hessian_matrix = calc.get_hessian(atoms=atoms)
    d1, num_atoms, dim = hessian_matrix.shape
    Cmat = hessian_matrix.reshape((d1, d1)) # in the unit of eV/A^2
    
    if np.allclose(Cmat, Cmat.T) != True:
        raise ValueError("hessian matrix is not hermitian!")

    mass_vec = np.repeat(atoms.get_masses(), dim) # in the unit of amu
    Dmat = Cmat / np.sqrt(np.outer(mass_vec, mass_vec)) # in the unit of eV/(A^2*amu)
    
    return Dmat, mass_vec


####################################################################################################
def calculate_Dmat_eigs(Dmat,
                        mass_vec,
                        tol = 1e-6,
                        ):
    """
    Compute the eigenvalues and eigenvectors of a dynamic matrix (Dmat), and sort zero 
    frequencies to the top.
    
    Mathematical operations:
        D @ P = P @ W        
        D = P @ W @ P.T
        W = P.T @ W @ P
        
    Args:
    --------
    Dmat : ndarray, shape=(3N, 3N )  **** in the unit of eV/(A^2*amu) ****
            The input square matrix, which is expected to be diagonalizable. In the context
            of phonon calculations, this is often the dynamical matrix (Hessian matrix).
    mass_vec: ndarray, shape=(3N, )  **** in the unit of amu ****
            The vector of atomic masses.
    tol : float, optional
            The tolerance for zero frequencies. Default is 1e-4.
            
    Returns:
    --------
    wsquares : ndarray, shape=(3N, )  **** in the unit of eV/(A^2*amu) ****
            The sorted eigenvalues of Dmat, corresponding to the squared frequencies (w2) in
            phonon calculations.
    Pmat : ndarray, shape=(3N, 3N )
            The matrix of sorted eigenvectors, which serves as the transformation matrix between
            the original coordinate system and the phonon coordinate system.
    wfreqs : ndarray, shape=(N-3, )  **** in the unit of sqrt(eV/(A^2*amu)) ****
            The sorted non-zero frequencies (w) in phonon calculations.
    num_modes : int, shape=(N-3, )
            The number of non-zero frequencies.
    """
    
    dim = 3
    eig_vals, eig_vecs = np.linalg.eigh(Dmat)

    indices_zero = np.where(np.abs(eig_vals) <  tol)[0]
    indices_nonz = np.where(np.abs(eig_vals) >= tol)[0]

    sorted_indices = np.concatenate([indices_zero, indices_nonz])
    sorted_eig_vals = np.take(eig_vals, sorted_indices)
    sorted_eig_vecs = np.take(eig_vecs, sorted_indices, axis=1)
    # mass_vec = np.take(mass_vec, sorted_indices)
    
    wsquares = sorted_eig_vals
    
    Q2Rmat = np.einsum("b,ab->ab", 1/np.sqrt(mass_vec), sorted_eig_vecs.T) # trans_Q2R
    R2Qmat = np.einsum("a,ab->ab", np.sqrt(mass_vec),   sorted_eig_vecs)   # trans_R2Q
    
    num_zeros = len(indices_zero)
    num_modes = len(indices_nonz)
    wfreqs = np.sqrt(np.abs(wsquares))
    wfreqs = wfreqs[dim:]
    
    if num_zeros != 3:
        raise ValueError(f"The number of zero frequencies is not 3! It is {num_zeros}.")
    
    return Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes

####################################################################################################
def make_coordinate_transforms(positions_init,
                               box_lengths,
                               Q2Rmat,
                               R2Qmat,
                               coordinate_type = "phonon",
                               ):
    
    """
    Generates coordinate transformation functions based on the input coordinate type.
    
    Args:
    --------
    num_molecules (int): The number of molecules in the system.
    coordinate_type (str, optional): The type of input coordinates. Defaults to "phonon".
        - "phonon": phonon coordinates.
        - "atomic": atomic coordinates.

    Returns:
    --------
    tuple: A tuple containing two functions:
        - trans_Q2R (function): Transforms phonon (Q) coordinates to atomic coordinates (R).
        - trans_R2Q (function): Transforms atomic coordinates (R) to phonon (Q) coordinates.
            
    Note:
    --------
    In this function:
        - `L` refers to `box_lengths`.
        - `R0` refers to `positions_init`.
    """

    R0 = positions_init
    L = box_lengths
    num_atoms, dim = R0.shape
    num_modes = (num_atoms - 1) * 3

    if coordinate_type == "phonon":
        
        def trans_Q2R(Q):
            R0base, _ = jnp.split(R0, [1], axis=0)
            R0rela_flatten = (R0 - R0base).flatten()

            Q_flatten = jnp.concatenate((jnp.zeros((dim, )), Q.flatten()), axis=0)
            R_flatten = R0rela_flatten + Q_flatten @ Q2Rmat

            R = R_flatten.reshape(num_atoms, dim)
            
            Rbase, _ = jnp.split(R, [1], axis=0)
            R = R - Rbase + R0base
            R = R - L * jnp.floor(R/L)
            return R
        
        def trans_R2Q(R):
            R0base, _ = jnp.split(R0, [1], axis=0)
            R0rela = (R0 - R0base)
            
            Rbase, _ = jnp.split(R, [1], axis=0)
            Rrela  = (R - Rbase)
            
            U = Rrela - R0rela
            U = U - L*jnp.rint(U/L)
            Q_flatten = U.flatten() @ R2Qmat

            Q = Q_flatten[dim:].reshape(num_modes, 1)
            return Q
    
    elif coordinate_type == "displacement":
        
        raise ValueError("coordinate_type = 'atomic' is discarded!")
        def trans_Q2R(Q):
            R0base, _ = jnp.split(R0, [1], axis=0)
            R0rela_flatten = (R0 - R0base).flatten()

            Q_flatten = jnp.concatenate((jnp.zeros((dim, )), Q.flatten()), axis=0)
            R_flatten = R0rela_flatten + Q_flatten @ Q2Rmat

            R = R_flatten.reshape(num_atoms, dim)
            
            Rbase, _ = jnp.split(R, [1], axis=0)
            R = R - Rbase + R0base
            R = R - L * jnp.floor(R/L)
            return R
        
        def trans_R2Q(R):
            R0base, _ = jnp.split(R0, [1], axis=0)
            R0rela = (R0 - R0base)
            
            Rbase, _ = jnp.split(R, [1], axis=0)
            Rrela  = (R - Rbase)
            
            U = Rrela - R0rela
            U = U - L*jnp.rint(U/L)
            Q_flatten = U.flatten() @ R2Qmat

            Q = Q_flatten[dim:].reshape(num_modes, 1)
            return Q
    
    return trans_Q2R, trans_R2Q



####################################################################################################
if __name__ == '__main__':
    key = jax.random.key(42)
    isotope = "H2O"
    dim = 3
    mace_dtype = "float64"
    # mace_dtype = "float32"
    
    # mace_model_path = "macemodel/mace_iceX_l1x64r4.0.model"
    mace_model_path = "macemodel/mace_iceX_l1x128r4.0.model"
    calc = make_mace_calculator(mace_model_path, mace_dtype)
    
    
    ## ice08
    init_stru_path = "structures/relax/ice08c_n016_p40.00.vasp"
    
    # create ice crystal
    from crystal import create_ice_crystal
    atoms, box_lengths, positions_init, num_molecules, density = create_ice_crystal(init_stru_path, 
                                                                                    isotope = isotope,
                                                                                    )
    print("structure:", init_stru_path)
    print("mace model:", mace_model_path)
    print("box_lengths:", box_lengths)
    print("positions_init.shape (Angstrom):", positions_init.shape)
    print("num_molecules:", num_molecules)
    print("density (kg/m^3):", density)
    print("isotope:", isotope, f" [{atoms.get_masses()[0]}  {atoms.get_masses()[-1]}]")

    # calculate hessian matrix and eigenvalues
    Dmat, mass_vec = calculate_hessian_matrix(atoms, calc)
    Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes = calculate_Dmat_eigs(Dmat,
                                                                      mass_vec,
                                                                      tol = 1e-6,
                                                                      )
    print(np.allclose(Q2Rmat@R2Qmat, np.eye(3*num_molecules*dim)))
    # change wfreqs from sqrt(eV/(A^2*amu)) -> s^-1 -> cm^-1
    wfreqs_invcm = wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm
    
    get_imag = 0
    if get_imag:
        imag_wfreqs = np.sign(wsquares[dim:]) * np.sqrt(np.abs(wsquares[dim:]))
        imag_wfreqs_invcm = imag_wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm
    
    if 1:
        # plot lorentz distribution
        import matplotlib.pyplot as plt
        from utils import get_lorentz_hist
        
        # lx = np.linspace(-2000, 4000, 10000)
        lx = np.linspace(0, 4000, 10000)
        gamma = 20.0
        
        lx, ly = get_lorentz_hist(wfreqs_invcm, lx=lx, gamma=gamma)

        if get_imag:
            imag_lx, imag_ly = get_lorentz_hist(imag_wfreqs_invcm, lx=lx, gamma=gamma)
        
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(lx, ly,  "-", linewidth=1.0, label="energy")
        if get_imag:
            plt.plot(imag_lx, imag_ly,  "--", linewidth=1.0, label="imag freqs")
        
        plt.title(init_stru_path)
        plt.grid(True)
        plt.xlabel(r"frequencies (cm$^{-1}$)")
        plt.ylabel(r"density of states")
        plt.show()

    ### create coordinate transformation functions
    trans_Q2R, trans_R2Q = make_coordinate_transforms(positions_init, 
                                                      box_lengths, 
                                                      Q2Rmat,
                                                      R2Qmat,
                                                      coordinate_type = "phonon"
                                                      )

    key = jax.random.PRNGKey(42)
    phoncoords0 = 0.01 * jax.random.normal(key, shape=(num_modes, 1))
    atomcoords0 = trans_Q2R(phoncoords0)
    phoncoords1 = trans_R2Q(atomcoords0)
    atomcoords1 = trans_Q2R(phoncoords1)

    print(jnp.allclose(phoncoords0, phoncoords1))
    print(jnp.allclose(atomcoords0, atomcoords1))














####################################################################################################

    # #======== tests ========#
    # hessian_matrix = calc.get_hessian(atoms=atoms)
    # d1, num_atoms, dim = hessian_matrix.shape
    # Cmat = hessian_matrix.reshape((d1, d1)) # in the unit of eV/A^2
    
    # mass_vec = np.repeat(atoms.get_masses(), dim) # in the unit of amu
    # Dmat = Cmat / np.sqrt(np.outer(mass_vec, mass_vec)) # in the unit of eV/(A^2*amu)
    # eig_vals, eig_vecs = np.linalg.eigh(Dmat)
    
    # wsquares = eig_vals[dim:]
    # wfreqs = np.sqrt(np.abs(wsquares)) # in the unit of  sqrt(eV/(A^2*amu))
    # wfreqs_invcm = wfreqs * np.sqrt(units.eV_over_A2_amu_to_inv_s2) * units.inv_s1_to_invcm # in the unit of cm^-1
    
    # Dmat, mass_vec = calculate_hessian_matrix(atoms, calc)
    # Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes = calculate_Dmat_eigs(Dmat, mass_vec, tol = 1e-6)
    
    # if 1:
    #     # plot lorentz distribution
    #     from utils import get_lorentz_hist
    #     from units import units
        
    #     lx, ly = get_lorentz_hist(wfreqs_invcm, 
    #                               lx=np.linspace(0, 4000, 10000), 
    #                               gamma=20.0
    #                               )
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(6, 4), dpi=300)
    #     plt.plot(lx, ly,  "-", linewidth=1.0, label="energy")
    #     plt.grid(True)
    #     plt.xlabel(r"frequencies (cm$^{-1}$)")
    #     plt.ylabel(r"density of states")
    #     plt.show()
    
    # ### create coordinate transformation functions
    # trans_Q2R, trans_R2Q = make_coordinate_transforms(positions_init, 
    #                                                   box_lengths, 
    #                                                   Q2Rmat,
    #                                                   R2Qmat,
    #                                                   coordinate_type = "phonon"
    #                                                   )

    # key = jax.random.PRNGKey(42)
    # phoncoords0 = 0.01 * jax.random.normal(key, shape=(num_modes, 1))
    # atomcoords0 = trans_Q2R(phoncoords0)
    # phoncoords1 = trans_R2Q(atomcoords0)
    # atomcoords1 = trans_Q2R(phoncoords1)

    # print(jnp.allclose(phoncoords0, phoncoords1))
    # print(jnp.allclose(atomcoords0, atomcoords1))

    # dim = 3
    
    # hessian_matrix = calc.get_hessian(atoms=atoms)
    # d1, num_atoms, dim = hessian_matrix.shape
    # Cmat = hessian_matrix.reshape((d1, d1)) # in the unit of eV/A^2

    # mass_vec = np.repeat(atoms.get_masses(), dim) # in the unit of amu
    # Dmat = Cmat / np.sqrt(np.outer(mass_vec, mass_vec)) # in the unit of eV/(A^2*amu)
    
    # # transform:  eV/(A^2*amu)  ->  s^-2
    # transfactor_s2 = units.eV_2_J / (units.amu_2_kg * units.angstrom_2_m ** 2)
    # # transform:  s^-2  ->  cm^-2
    # transfactor_cm2 = 1 / ( (2 * np.pi * units.c * units.m_to_cm)**2 )
    
    # Dmat = Dmat * transfactor_s2 * transfactor_cm2
    
    # eig_vals, eig_vecs = np.linalg.eigh(Dmat)
    # wfreqs = np.sqrt(eig_vals)
    
    # Dmat, mass_vec = calculate_hessian_matrix(atoms, calc, num_molecules, mass_H, mass_O)
    # Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes = calculate_Dmat_eigs(Dmat, mass_vec, tol = 1e-6)

    # print(wfreqs)
    # print(wfreqs*units.eV_2_cminv)

    # if 1:
    #     # plot lorentz distribution
    #     from utils import get_lorentz_hist
    #     from units import units
        
    #     lx, ly = get_lorentz_hist(wfreqs * units.eV_2_cminv, 
    #                               lx=np.linspace(0, 20000, 10000), 
    #                               gamma=100.0
    #                               )
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(6, 4), dpi=300)
    #     plt.plot(lx, ly,  "-", linewidth=1.0, label="energy")
    #     plt.grid(True)
    #     plt.xlabel(r"frequencies (cm^-1)")
    #     plt.ylabel(r"density of states")
    #     plt.show()

    
    # ####
    # hessian_matrix = calc.get_hessian(atoms=atoms)
    # d1, num_atoms, dim = hessian_matrix.shape
    # hessian_matrix = hessian_matrix.reshape((d1, d1)) # in the unit of eV/A^2
    
    # Cmat = hessian_matrix
    # num_O = num_molecules
    # num_H = num_molecules * 2
    # mass_vec = np.array([mass_H] * num_H * dim + [mass_O] * num_O * dim) # in the unit of amu
    # mass_mat_inv = 1 / np.sqrt(np.outer(mass_vec, mass_vec)) # in the unit of 1/amu

    
    # Dmat, mass_vec = calculate_hessian_matrix(atoms, calc, num_molecules, mass_H, mass_O)
    # Q2Rmat, R2Qmat, wsquares, wfreqs, num_modes = calculate_Dmat_eigs(Dmat, mass_vec, tol = 1e-6)
    
    # if 1:
    #     # plot lorentz distribution
    #     from utils import get_lorentz_hist
    #     from units import units
        
    #     lx, ly = get_lorentz_hist(wfreqs * units.eV_2_cminv, 
    #                               lx=np.linspace(0, 50000, 10000), 
    #                               gamma=10.0
    #                               )
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(6, 4), dpi=300)
    #     plt.plot(lx, ly,  "-", linewidth=1.0, label="energy")
    #     plt.grid(True)
    #     plt.xlabel(r"frequencies (cm^-1)")
    #     plt.ylabel(r"density of states")
    #     plt.show()

    # ### create coordinate transformation functions
    # trans_Q2R, trans_R2Q = make_coordinate_transforms(positions_init, 
    #                                                   box_lengths, 
    #                                                   Q2Rmat,
    #                                                   R2Qmat,
    #                                                   coordinate_type = "phonon"
    #                                                   )

    # key = jax.random.PRNGKey(42)
    # phoncoords0 = 1.0 * jax.random.normal(key, shape=(num_modes, 1))
    # atomcoords0 = trans_Q2R(phoncoords0)
    # phoncoords1 = trans_R2Q(atomcoords0)
    # atomcoords1 = trans_Q2R(phoncoords1)

    # print(jnp.allclose(phoncoords0, phoncoords1))
    # print(jnp.allclose(atomcoords0, atomcoords1))
    

# def make_coordinate_transforms(num_molecules,
#                                coordinate_type = "phonon",
#                                ):
    
#     """
#     Generates coordinate transformation functions based on the input coordinate type.
    
#     Args:
#     --------
#     num_molecules (int): The number of molecules in the system.
#     coordinate_type (str, optional): The type of input coordinates. Defaults to "phonon".
#         - "phonon": Phonon coordinates.
#         - "atomic": Displacement coordinates.

#     Returns:
#     --------
#     tuple: A tuple containing two functions:
#         - trans_Q2R (function): Transforms phonon (Q) coordinates to real space coordinates (R).
#         - trans_R2Q (function): Transforms real space coordinates (R) to phonon (Q) coordinates.
            
#     Note:
#     --------
#     In this function:
#         - `L` refers to `box_lengths`.
#         - `R0` refers to `positions_init`.
#     """
    
#     dim = 3
#     num_atoms = num_molecules * 3
#     num_modes = (num_molecules * 3 - 1) * dim

#     if coordinate_type == "phonon":
        
#         ## trans_Q2R(phoncoords, positions_init, box_lengths, Pmat)
#         def trans_Q2R(Q, R0, L, Pmat):
#             R0base, _ = jnp.split(R0, [1], axis=0)
#             R0rela_flatten = (R0 - R0base).flatten()

#             Q_flatten = jnp.concatenate((jnp.zeros((dim, )), Q.flatten()), axis=0)
#             R_flatten = R0rela_flatten + Q_flatten @ Pmat.T
#             R = R_flatten.reshape(num_atoms, dim)
            
#             Rbase, _ = jnp.split(R, [1], axis=0)
#             R = R - Rbase + R0base
#             R = R - L * jnp.floor(R/L)
#             return R
        
#         ## trans_R2Q(positions, positions_init, box_lengths, Pmat)
#         def trans_R2Q(R, R0, L, Pmat):
#             R0base, _ = jnp.split(R0, [1], axis=0)
#             R0rela = (R0 - R0base)
            
#             Rbase, _ = jnp.split(R, [1], axis=0)
#             Rrela  = (R - Rbase)
            
#             U = Rrela - R0rela
#             U = U - L*jnp.rint(U/L)
#             Q_flatten = U.flatten() @ Pmat
#             Q = Q_flatten[dim:].reshape(num_modes, 1)
#             return Q
    
#     elif coordinate_type == "atomic":
        
#         raise ValueError("coordinate_type = 'atomic' is discarded!")
    
#     return trans_Q2R, trans_R2Q

