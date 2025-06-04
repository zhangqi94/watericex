
####################################################################################################

class units:
    """
    Atomic units to SI units conversion factors
    """
    # Energy conversion
    eV_2_J = 1.602176634e-19  # 1 eV to Joules

    # Mass conversion
    amu_2_kg = 1.660539066e-27  # 1 amu to kilograms

    # Length conversion
    angstrom_2_m = 1e-10  # 1 angstrom to meters
    
    # Planck's constant
    h_in_J = 6.62607015e-34  # Planck's constant in J·s
    h_in_eV = 4.135667696e-15  # Planck's constant in eV·s

    # Pressure conversion
    eVangstrom_2_GPa  = 160.21766208  # eV/angstrom^3 to GPa
    Kelvin_2_GPa = 0.01380649  # K/angstrom^3 to GPa

    # Energy-temperature conversion
    Kelvin_2_meV = 0.08617333262145  # K to meV
    Kelvin_2_eV = 8.617333262145e-5  # K to eV
    Kelvin_2_J = 1.38064852e-23  # K to Joules
    meV_2_eV = 1e-3  # meV to eV
    
    # Energy to wavenumber conversion
    # eV to cm^-1
    # mu = E (in J) / (h * c * 100)
    eV_2_cminv = 8065.543937349212  # eV to cm^-1
    meV_2_cminv = 8.065543937349212  # meV to cm^-1
    
    # Speed of light and length conversion
    c = 299792458  # Speed of light in m/s
    m_to_cm = 100  # Meters to centimeters

    # Vibrational frequency conversion
    eV_over_A2_amu_to_inv_s2 = 9.648533219151616e+27  # eV/(A^2·amu) to s^-2
    inv_s1_to_invcm = 5.308837458876145e-12  # s^-1 to cm^-1

    # Reduced Planck's constant
    hbar_in_eVamuA = 0.06465415135263675  # Reduced Planck's constant in (eV·amu)^(1/2)·A

"""
    eV/(A^2 * amu) to s^-2
        1.602176634e-19 / ((1e-10)**2 * 1.660539066e-27)
    s^-1 to cm^-1
        1 / (100 * 299792458 * 2 * np.pi)
        
    hbar = 6.62607015e-34 / (2*np.pi)  J * s
        -> (6.62607015e-34 / (2*np.pi)) / np.sqrt(1.602176634e-19 * 1.660539066e-27 * 1e-10**2)
         = 0.06465415135263675 (eV * amu)^(1/2) * A
"""
    
####################################################################################################
# Atomic masses in amu
mass_H_list = [
    1.007825031898,  # Hydrogen-1
    2.014101777844,  # Hydrogen-2 (Deuterium)
    3.016049281320,  # Hydrogen-3 (Tritium)
]

mass_O_list = [
    15.994914619257,  # Oxygen-16
    16.999131755953,  # Oxygen-17
    17.999159612136,  # Oxygen-18
]

####################################################################################################

if __name__ == '__main__':
    # Print conversion factors
    1 * units.amu_2_kg 
    
    # print("amu_2_kg:", units.amu_2_kg)
    # print("angstrom_2_m:", units.angstrom_2_m)
    # print("eV_2_J:", units.eV_2_J)

    # # Print Planck's constants
    # print("h_in_J:", units.h_in_J)
    # print("h_in_eV:", units.h_in_eV)

    # # Print pressure conversion factors
    # print("meVangstrom_2_GPa:", units.meVangstrom_2_GPa)
    # print("Kelvin_2_GPa:", units.Kelvin_2_GPa)
    # print("Kelvin_2_meV:", units.Kelvin_2_meV)

    # # Print atomic mass lists
    # print("mass_H_list:", mass_H_list)
    # print("mass_O_list:", mass_O_list)


