import ase
import ase.io
from ase import Atoms
import numpy as np

import matplotlib.pyplot as plt
from anastrutools import load_energy_data


####################################################################################################
if 0:
    # dft_func = "pbe"
    dft_func = "scan"

    a = 2.909331
    a = 2.855228
    a = 2.744977
    a = 2.717028
    a = 2.624119

    list_d = np.linspace(0, 0.55, 111)

    f = open(f"data_dft_functional/{dft_func}_a_{a:.6f}.txt", "w")

    for i in range(len(list_d)):

        d = list_d[i]
        file_name = f"/home/zq/zqdata/iceX/data_dft_functional/{dft_func}/a_{a:.6f}_d_{d:.6f}/OUTCAR"    
        atoms = ase.io.read(file_name)
        e = atoms.get_potential_energy()
        print(f"{a:.6f}  {d:.6f}  {e:.6f}")
        f.write("%.6f  %.6f  %.12f\n" %(a, d, e))

    f.close()

####################################################################################################
if 1:
    
    colors = np.array([[100, 149, 237],
                    [8, 81, 156],
                    [50, 205, 50],
                    [34, 139, 34],
                    [247, 144, 61],
                    [214, 39, 40],
                    [244, 114, 117],
                    [139, 0, 0],
                    ],
                    )/255

    plt.figure(figsize=(8, 4), dpi=300)
    for i in range(len(colors)):
        y = np.arange(0, 1, 0.01)
        x = np.ones_like(y) * i
        plt.plot(x, y, "-", linewidth = 20, color = colors[i], label = str(i) )
    plt.xlim([-1, 10])
    plt.legend()


if 1:
    
    f1 = "/home/zq/zqcodeml/waterice/anastrus/data_dft_functional/pbe_a_2.624119.txt"
    f2 = "/home/zq/zqcodeml/waterice/anastrus/data_dft_functional/pbe_a_2.744977.txt"
    f3 = "/home/zq/zqcodeml/waterice/anastrus/data_dft_functional/pbe_a_2.909331.txt"
    g1 = "/home/zq/zqcodeml/waterice/anastrus/data_dft_functional/scan_a_2.624119.txt"
    g2 = "/home/zq/zqcodeml/waterice/anastrus/data_dft_functional/scan_a_2.744977.txt"
    g3 = "/home/zq/zqcodeml/waterice/anastrus/data_dft_functional/scan_a_2.909331.txt"
    
    ## around 120GPa, 70GPa, 30GPa
    Ls = np.array([2.624119, 2.744977, 2.909331])
    Vols = (Ls*2)**3/16
    print(Vols)

    plt.figure(figsize=(5.5, 3.9), dpi=300)
    
    x, y = load_energy_data(f1)
    plt.plot(x, y, linestyle='--', linewidth=1, color=colors[0], 
             label = r"$\Omega = 9.035~\AA^3/\text{H}_2\text{O}$, PBE", zorder=10)
    print(np.min(y))
    x, y = load_energy_data(g1)
    plt.plot(x, y, linestyle='-', linewidth=1, color=colors[1], 
             label = r"$\Omega = 9.035~\AA^3/\text{H}_2\text{O}$, SCAN", zorder=9)
    print(np.min(y))
    
    x, y = load_energy_data(f2)
    plt.plot(x, y, linestyle='--', linewidth=1, color=colors[2], 
             label = r"$\Omega = 10.035~\AA^3/\text{H}_2\text{O}$, PBE", zorder=8)
    print(np.min(y))
    x, y = load_energy_data(g2)
    plt.plot(x, y, linestyle='-', linewidth=1, color=colors[3], 
            label = r"$\Omega = 10.035~\AA^3/\text{H}_2\text{O}$, SCAN", zorder=7)
    print(np.min(y))
    
    x, y = load_energy_data(f3)
    plt.plot(x, y, linestyle='--', linewidth=1, color=colors[4], 
             label = r"$\Omega = 12.312~\AA^3/\text{H}_2\text{O}$, PBE", zorder=6)
    print(np.min(y))
    x, y = load_energy_data(g3)
    plt.plot(x, y, linestyle='-', linewidth=1, color=colors[5], 
             label = r"$\Omega = 12.313~\AA^3/\text{H}_2\text{O}$, SCAN", zorder=5)
    print(np.min(y))
    
    plt.xlabel(r"$d_{\text{OH}}-\frac{1}{2}d_{\text{OO}}~(\AA)$", fontsize=12.0)
    plt.ylabel(r"$V$ (eV)", fontsize=12.0)
    # plt.legend(fontsize=9)
    plt.legend(fontsize=9, frameon=True, edgecolor='gray', loc='best', fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=9)

    plt.xlim([-0.45, 0.45])
    plt.ylim([-0.5, 1])
    plt.grid(1)
    plt.tight_layout()
    plt.savefig("fig_dft.pdf", format='pdf')
    plt.show()




