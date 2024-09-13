In these files we include the basic functions to implement the tangent fermions method to simulate chiral, massless Dirac fermions on a lattice without fermion doubling together with some exmples. The manuscript developing the theory behind it can be found at arXiv:2302.12793.

The libraries needed to run the code are numpy, scipy, math and matplotlib.pyplot.

Any system is defined by the following parameters. Units are chosen so that lattice constant is ds = 1, Fermi velocity is c = 1, reduced Plack constant is hbar = 1 and e = 1.

SYSTEM PARAMETERS:

    - Nx: int. Length of the system
    - Ny: int. Width of the system
    - B1: float. Average magtnetic field on one region of the system (if it is not zero, another region with opposite magnetic field must exist to make the total flux zero)
    - N1: int. Width of the region with average field B1
    - N2: int. Width of the region with opposite magnetic field
    - d1: int. Width of the zero field region separating the previous two
    - dt: float. Time step
    - potential: function of x and y arrays. Function representing the electrostatic potential profile.
    - mass: function of x and y arrays. Function representing the mass profile.
    - ky: float. Phase spectifying the periodic boundary conditions in y direction
    - kx: float. Phase spectifying the periodic boundary conditions in x direction
    - disorder: float. Disorder strength for magnetic field. A random value in (-disorder/2, disorder/2) is added to the uniform flux on each plaquette.
    
In addition, some functions require the parameters characterising a gaussian wave packet to be used as initial state.

INITIAL STATE PARAMETERS:

    - k: float. Absolute value of the mean momentum of the packet
    - phase: float. Angle of mean momentum with respect to the x axis
    - sigma: float. Standard deviation of the gaussian wave packet
    - centre: 2-ple of floats. center of the wave packet in space.


The file tangent_fermions.py contains the following functions.

    - make_potential: returns an array corresponding to the electrostatic potential given function
    - make_mass: returns an array corresponding to the mass (sigma_z coefficient) given function
    - make_fluxes: returns an array corresponding to the magnetic flux through each plaquette of the system given a total magnetic field and a disorder strength.
            Since the total flux must be zero the function takes the size of the region with positive (N1), negative (N2) and zero field region (d1).
    - vector_potential: returns Peierls phases given a fluxes array
    - make_initial_gaussian: produces an array corresponding to the initial wavefunction: a gaussian wave packet that is specified by the size of the system (Nx, Ny) and:
            -centre: 2-tuple. The centre of the wave packet
            -sigma: float. The standard deviation of the gaussian.
            -k: float. average wavenumber of the gaussian (absolute value).
            -phase; float (mod 2*pi). angle of the average wavenumber with the x axis
    - operators_real: returns the operators Phi, P and H in real space as described in arXiv:2302.12793 given the parameters of the system
    - tangent_fermions_evolution_real: Evolves an initial wave packet using the sparse LU decomposition method (in real space). Returns the wavefunction at all time steps plus the lu decomposition for reuse.
    - operators_reciprocal: Builds the arrays needed to implement time evolution using the split-operator approach. (Only supports kx = ky = 0).
    - tangent_fermion_evolution_reciprocal: Evolves an initial wave packet using the split operator method (in reciprocal space). (Only supports kx = ky = 0).
    - make_bands_x: calculates and plots bands in x direcion for a given system.

The jupyter notebooks show how to use these functions to calculate spectra, bands and densities of states. They also show how to evolve states in time and reproduce some of the figures of arXiv:2302.12793.

NOTEBOOKS

- spectrum_and_eigenstates.ipynb: In this notebook we show how to use the code in tangent_fermions.py to generate and solve the generalised eigenvalue equation 3.21 in arXiv:2302.12793. Tangent fermions have poles in the edges of the Brillouin zone, so we would like to avoid them. Two possible ways to do that are to pick an odd number of unit cells with periodic boundary conditions (like we are doing in this case: Nx = Ny = 101 and kx = ky = 0) or to pick an even number of cells and antiperiodic boundary conditions. We find the spectrum and band structure of systems with and without magnetic field, with and without disorder.

- time_evolution.ipynb: In this notebook we show how to use the code in tangent_fermions.py to evolve wave packets as explained in arXiv:2302.12793. First, we evolve a gaussian wave packet moving towards a Kein step with some angle in absence of magnetic field. We can use the split operator approach according to equation 5.5 in arXiv:2302.12793. We repeat the same simulation using the sparse LU approach (equation 7.4). The latter method can also be used to evolve states in presence of a magnetic field, which we do next. Finally, we use these functions to reproduce the Klein barrier of figure 6 in arXiv:2302.12793.

- density_of_states.ipynb: In this notebook we show how to use the code in tangent_fermions.py to calculate DOS in a system. We do so by reproducing a figure analogous to figure 10 of arXiv:2302.12793 (we use a smaller system in order o make it leass time consuming, but we also include the parameters to run obtain figure 10.)

- majorana_metal.ipynb: In this notebook we show how to use the code in tangent_fermions.py produce the phase diagram of figure 11 of arXiv:2302.12793 (again we start by calculating the phase diagram with a coarser sampling for efficiency reasons, but we also include the parameters to reproduce the original figure).
