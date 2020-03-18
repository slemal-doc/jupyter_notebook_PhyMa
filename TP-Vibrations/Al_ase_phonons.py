from ase.build import bulk
from ase.calculators.emt import EMT
from ase.phonons import Phonons
from ipywidgets import FloatSlider, IntSlider, interact, interact_manual, fixed

def calculate_phonons(x):
    # Setup crystal and EMT calculator
    atoms = bulk('Al', 'fcc', a=x)#4.05)

    # Phonon calculator
    N = 7
    ph = Phonons(atoms, EMT(), supercell=(N, N, N), delta=0.05)
    ph.run()

    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)
    ph.clean()

    path = atoms.cell.bandpath('GXULGK', npoints=100)
    bs = ph.get_band_structure(path)

    dos = ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=100, width=1e-3)
    
    forces = ph.get_force_constant()
    print (forces)

    # Plot the band structure and DOS:
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(8, 4), dpi=300)
    ax = fig.add_axes([.12, .07, .67, .85])

    emax = 0.035
    
    bs.plot(ax=ax, emin=-0.01, emax=emax)

    dosax = fig.add_axes([.8, .07, .17, .85])
    dosax.fill_between(dos.weights[0], dos.energy, y2=0, color='grey',
                       edgecolor='k', lw=1)

    dosax.set_ylim(-0.01, emax)
    dosax.set_yticks([])
    dosax.set_xticks([])
    dosax.set_xlabel("DOS", fontsize=18)

    fig.savefig('Al_phonon.png')
    return

interact_manual(calculate_phonons, x=FloatSlider(value=4.05, min=0, max=10, description='a (Angstrom)'));
