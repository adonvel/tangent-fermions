import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.fft import fft2, ifft2
from scipy.sparse import csr_matrix, csc_matrix, linalg as sla
from math import pi

sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

# Units are chosen so that lattice constant is ds = 1, velocity is c = 1, reduced Plack constant is hbar = 1 and e = 1.

def make_potential(parameters, plot = False):
    '''Produces an array potential_array[y,x] = potential(x,y).'''
    
    Nx = parameters['Nx'] #Odd
    Ny = parameters['Ny'] #Odd
    potential_function = parameters['potential'] #Must be applicable to arrays

    x, y = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny), sparse=False, indexing='xy')
    potential_array = potential_function(x,y)
    
    if plot:
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.imshow(potential_array, origin = 'lower')
        ax.set_xlabel(r'$x$ (a)', fontsize = 14)
        ax.set_ylabel(r'$y$ (a)', fontsize = 14)
        ax.set_title(r'Potential', fontsize = 20)
        
    return potential_array

def make_mass(parameters, plot = False):
    '''Produces an array mass_array[y,x] = mass(x,y).'''
    
    Nx = parameters['Nx'] #Odd
    Ny = parameters['Ny'] #Odd
    mass_function = parameters['mass'] #Must be applicable to arrays

    x, y = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny), sparse=False, indexing='xy')
    mass_array = mass_function(x,y)
    
    if plot:
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.imshow(mass_array, origin = 'lower')
        ax.set_xlabel(r'$x$ (a)', fontsize = 14)
        ax.set_ylabel(r'$y$ (a)', fontsize = 14)
        ax.set_title(r'Mass', fontsize = 20)
        
    return mass_array


def make_fluxes(parameters, plot = False):
        '''Produces array of magnetic fluxes for 2 regions with opposite average field.'''

        Nx = parameters['Nx']
        Ny = parameters['Ny']
        N1 = parameters['N1']
        d1 = parameters['d1']
        N2 = parameters['N2']
        B1 = parameters['B1']  #Magnetic field in the first region
        disorder = parameters['disorder']
        
        if B1 == 0:
            B2 = 0
        elif N2 == 0:
            print('Warning: N2 must be non-zero for the total flux through the system to vanish.')
        else:
            B2 = B1*N1/N2
            
        d2 = Ny-N1-N2-d1
        

        fluxes = np.zeros((Ny, Nx))
        for x in range(Nx):
            for y in range(Ny):
                if y < N1:
                    fluxes[y,x] = B1 + disorder*(np.random.rand()-0.5)
                elif y < N1 + d1:
                    fluxes[y,x] = 0
                elif y < N1 + d1 + N2:
                    fluxes[y,x] = -B2 + disorder*(np.random.rand()-0.5)
                else:
                    fluxes[y,x] = 0

        fluxes = fluxes-np.sum(fluxes)/(Nx*Ny)
    
        if plot:
            fig = plt.figure(figsize = (7,7))
            ax = fig.add_subplot(111)
            ax.imshow(fluxes, origin = 'lower', cmap = 'bwr',vmax = max(np.amax(fluxes),-np.amin(fluxes)),vmin=min(-np.amax(fluxes),np.amin(fluxes)))
            ax.set_xlabel(r'$x$ (a)', fontsize = 14)
            ax.set_ylabel(r'$y$ (a)', fontsize = 14)
            ax.set_title(r'Magnetic field', fontsize = 20)
            
        return fluxes
    

def vector_potential(parameters, fluxes):
    'Obtain Peierls phases from fluxes through each lattice cell.'
    
    Nx = parameters['Nx']
    Ny = parameters['Ny']
    
    
    # We are assuming a gauge in which A is zero along all edges except at the top one,
    # where it is equal -total_flux/Nx. This should produce the ordinary magnetic translations exp(1j*B*Ny*x).

    def index(direction, i,j):
        if i < Nx and j < Ny:              # first all the hoppings in the unit cell
            idx = (j*Nx+i)*2 + direction 
        elif j == Ny:                       # then all hoppings on the top edge
            idx = 2*Nx*Ny+i
        elif i == Nx:                       # then all hoppings on the right edge
            idx = 2*Nx*Ny+Nx+j
        return idx

    row = []
    col = []
    data = []
    rhs = []

    row_i = 0

    # Rotational equations
    for i in range(Nx):
        for j in range(Ny):
            if not (i==Nx//2 and j==Ny//2): #skip one (one of them is linearly dependent)
                row += [row_i,row_i, row_i, row_i]
                col += [index(0,i,j),index(1,i+1,j),index(0,i,j+1),index(1,i,j)]
                data += [1,1,-1,-1]
                rhs += [fluxes[j,i]]
                row_i += 1

    # Divergence equations (not at the edges): Coulomb gauge 
    for i in range(1,Nx):
        for j in range(1,Ny):
            row += [row_i, row_i, row_i, row_i]
            col += [index(0,i,j), index(1,i,j), index(0,i-1,j), index(1,i,j-1)]
            data += [1,1,-1,-1]
            rhs += [0]
            row_i += 1

    #Fix the value of A at the edges (allowed by gauge freedom)
    for i in range(Nx): #bottom edge = 0
        row += [row_i]
        col+= [index(0,i,0)]
        data += [1]
        rhs += [0]
        row_i += 1

    for j in range(Ny): #left edge = 0
        row += [row_i]
        col+= [index(1,0,j)]
        data += [1]
        rhs += [0]
        row_i += 1

    for i in range(Nx): #top edge = -total_flux/Nx =0 in this case
        row += [row_i]
        col+= [index(0,i,Ny)]
        data += [1]
        rhs += [0]
        row_i += 1

    for j in range(Ny): #right edge = 0
        row += [row_i]
        col += [index(1,Nx,j)]
        data += [1]
        rhs += [0]
        row_i += 1
        

    equations = csr_matrix((data, (row, col)), shape=(2*Nx*Ny+Nx+Ny, 2*Nx*Ny+Nx+Ny))
    vector_potential = sla.spsolve(equations, rhs)
    vector_potential = vector_potential[:2*Nx*Ny].reshape(Ny,Nx,2)
    a_e = vector_potential[:,:,0]
    a_n = vector_potential[:,:,1]
    
    a_e = a_e - np.average(a_e)
    a_n = a_n - np.average(a_n)
    
    return a_e, a_n


def make_initial_gaussian(parameters, plot = False):
    '''Produces gaussian initial state array.'''
    
    Nx = parameters['Nx']            #Number of unit cells in x direction
    Ny = parameters['Ny']            #Number of unit cells in y direction
    k = parameters['k']              #Wave number in units of ds^{-1}
    phase = parameters['phase']      #Angle of momentum with the x axis
    sigma = parameters['sigma']      #Standard deviation of the wave packet in units of ds
    centre = parameters['centre']    #Centre of the wave packet in units of ds.
    
    x, y = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny), sparse=False, indexing='xy')
    
    x = (x - centre[0] + Nx/2)%Nx-Nx/2
    y = (y - centre[1] + Ny/2)%Ny-Ny/2
    
    
    wavefunction = np.array([np.exp(-1j*phase/2)*(np.exp(-(x**2+y**2)/(2*(sigma)**2))
                         * np.exp(1j*k*np.cos(phase)*x)
                        * np.exp(1j*k*np.sin(phase)*y)),np.exp(1j*phase/2)
                        *(np.exp(-(x**2+y**2)/(2*(sigma)**2))
                         * np.exp(1j*k*np.cos(phase)*x)
                        * np.exp(1j*k*np.sin(phase)*y))])
            
    wavefunction = wavefunction/np.sqrt(np.sum(np.abs(wavefunction)**2))
    
    if plot:
        probability = np.sum(np.abs(wavefunction)**2,axis = 0)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.imshow(probability, origin = 'lower')
        ax.set_xlabel(r'$x$ (a)', fontsize = 14)
        ax.set_ylabel(r'$y$ (a)', fontsize = 14)
        ax.set_title(r'Initial wave function', fontsize = 20)
    
    return wavefunction

def operators_real(parameters, plot_potential = False, plot_mass = False, plot_mag_field = False):
    '''Returns operators Phi, H and P.'''
    
    Nx = parameters['Nx']     # Number of unit cells in x direction (should be odd)
    Ny = parameters['Ny']     # Number of unit cells in y direction (should be odd)
    kx = parameters['kx']     # BC phase in x direction in units ds^(-1)
    ky = parameters['ky']     # BC phase in y direction in units ds^(-1)
    
    #Generate Peierls phases
    #np.random.seed(0)
    if parameters['B1'] == 0:
        a_e = np.zeros((Ny,Nx))
        a_n = np.zeros((Ny,Nx))
    else:
        fluxes = make_fluxes(parameters, plot = plot_mag_field)
        a_e, a_n = vector_potential(parameters,fluxes)
        
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    
    
    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
        
        #Peierls phases
        p_e = np.exp(1j*(a_e[y,x]))
        p_n = np.exp(1j*(a_n[y,x]))
        
        #Standard translations
        trs_e = np.exp(-(1j*kx*Nx)*((x+1)//Nx))
        trs_n = np.exp(-(1j*ky*Ny)*((y+1)//Ny))
        
        #Total phases
        phase_e = p_e*trs_e
        phase_n = p_n*trs_n
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e]
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n]
        

    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2

    potential_array = make_potential(parameters, plot = plot_potential).flatten()    
    pot = scipy.sparse.spdiags(potential_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    V = scipy.sparse.kron(csc_matrix(sigma_0), pot, format = "csc")    
    
    mass_array = make_mass(parameters, plot = plot_mass).flatten()    
    mass = scipy.sparse.spdiags(mass_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")
    P = (Phi.H)@Phi
    H = H_0 + Phi.H@V@Phi + Phi.H@M@Phi

    return Phi, H, P 


def tangent_fermions_evolution_real(parameters, known_lu = False, plot_potential = False, plot_mass = False, plot_mag_field = False, plot_init_state = False):
    '''Evolution in real space (sparse LU) given some initial state and parameters.
    Returns the wavefunction at all time steps plus the lu decomposition for reuse.'''
    
    Nx = parameters['Nx']      # Number of unit cells in x direction (should be odd)
    Ny = parameters['Ny']      # Number of unit cells in y direction (should be odd)
    dt = parameters['dt']      # Time step in units ds/v
    Nt = parameters['Nt']      # Number of time steps
    
    # Build operators necessary for time evolution
    Phi, H, P = operators_real(parameters, plot_potential = plot_potential, plot_mass = plot_mass, plot_mag_field = plot_mag_field)
    
    # Final matrices needed for implementing time evolution LHS @ psi(t+dt) = RHS @ psi(t)
    LHS = P + 1j*H/2*dt    # left-hand-side
    RHS = P - 1j*H/2*dt    # right-hand-side
    
    # Do LU decomposition
    if not known_lu:
        lu = sla.splu(LHS)
    else:
        lu = known_lu
        

    state = np.zeros((Nt, 2*Nx*Ny), dtype= complex) # This is psi
    wavefunction = np.zeros((Nt, 2*Nx*Ny), dtype= complex) # This is phi acting on psi
    
    state[0] = make_initial_gaussian(parameters, plot = plot_init_state).flatten()
    wavefunction[0] = Phi@(state[0])

    for t in range(Nt-1):
        state[t+1] = lu.solve(RHS@state[t])
        wavefunction[t] = Phi@(state[t])
        
    return np.reshape(wavefunction, (Nt,2,Ny,Nx)), lu


def operators_reciprocal(parameters):
    '''Builds the arrays to implement time evolution using the split-operator approach. '''
    
    Nx = parameters['Nx']            #Number of unit cells in x direction
    Ny = parameters['Ny']            #Number of unit cells in y direction
    dt = parameters['dt']            #Time step in units of ds/v
    
    kx, ky = np.meshgrid(np.linspace(-pi,pi,Nx,endpoint = False), np.linspace(-pi,pi,Ny,endpoint = False), sparse=False, indexing='xy')
    kx = np.roll(kx,Nx//2,axis = 1)
    ky = np.roll(ky,Ny//2,axis = 0)

    tank = 2*np.sqrt(np.tan(ky/2)**2+np.tan(kx/2)**2)
    
    diagonal_term = (1-(dt/2*tank)**2)/(1+(dt/2*tank)**2)
    spin_mix_term_u = -2j*dt*(np.tan(kx/2)-1j*np.tan(ky/2))/(1+(dt/2*tank)**2)
    spin_mix_term_v = -2j*dt*(np.tan(kx/2)+1j*np.tan(ky/2))/(1+(dt/2*tank)**2)
    
    return diagonal_term, spin_mix_term_u, spin_mix_term_v


def tangent_fermions_evolution_reciprocal(parameters, plot_potential = False, plot_mass = False, plot_init_state = False):
    ''' Evolves the initial state using the split-operator scheme with periodic boundary conditions.'''
    
    Nx = parameters['Nx']            #Number of unit cells in x direction
    Ny = parameters['Ny']            #Number of unit cells in y direction
    Nt = parameters['Nt']            #Number of time steps
    dt = parameters['dt']            #Time step in units of ds/v

    #Define evolution function
    diagonal_term, spin_mix_term_u, spin_mix_term_v = operators_reciprocal(parameters)
    potential = make_potential(parameters, plot = plot_potential)
    mass = make_mass(parameters, plot = plot_mass)
                        
                        
    def evolve(state):
        
        u_fourier = fft2(np.exp(-1j*potential*dt/2)*np.exp(-1j*mass*dt/2)*state[0])
        v_fourier = fft2(np.exp(-1j*potential*dt/2)*np.exp(1j*mass*dt/2)*state[1])

        u_fourier_evolved = diagonal_term * u_fourier + spin_mix_term_u * v_fourier
        v_fourier_evolved = spin_mix_term_v * u_fourier + diagonal_term * v_fourier

        u_next_step = np.exp(-1j*potential*dt/2)*np.exp(-1j*mass*dt/2)*ifft2(u_fourier_evolved)
        v_next_step = np.exp(-1j*potential*dt/2)*np.exp(1j*mass*dt/2)*ifft2(v_fourier_evolved)

        return np.array([u_next_step, v_next_step])
    
    #Produce initial state
    state = np.zeros((Nt,2,Ny,Nx),dtype = complex)
    state[0] = make_initial_gaussian(parameters, plot = plot_init_state)

        
    #Evolve in time
    for t in range(Nt-1):       
        state[t+1] = evolve(state[t])
    
    return state


def make_bands_x(parameters, kmin = -3, kmax = 3,  number_of_points = 101, number_of_bands = int(20), save_bands = False, plot_bands = True):
    '''Calculate and plot bands in x direction.'''
    
    #Generate Peierls phases
    np.random.seed(0)
    fluxes = make_fluxes(parameters)
    a_e, a_n = vector_potential(parameters,fluxes)
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n

    #Solve generalised eigenproblem for each kx.
    k_points = np.linspace(kmin/parameters['Nx'],kmax/parameters['Nx'],number_of_points)
    bands = np.zeros((number_of_points, number_of_bands))

    for i,kx in enumerate(k_points):

        parameters['kx'] = kx

        Phi, H, P = operators_real(parameters)
        eigenvalues = sla.eigsh(H, M=P, k = number_of_bands, tol = 0, sigma = 0.000001, which = 'LM',return_eigenvectors = False)

        bands[i] = np.sort(eigenvalues)
        
    if save_bands:
        np.save("bands_"+"_Ny_"+str(parameters['Ny'])+"_Nx"+str(parameters['Nx']) +"_B"+str(int(100*parameters['B1']))+"_disorder"+str(int(100*parameters['disorder']))+"_numbands"+str(number_of_bands), bands)
    
    #Plot
    if plot_bands:
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)

        for band in range(number_of_bands):
            ax.scatter(k_points,bands[:,band], c = 'k',s = 0.2)

        ax.set_xlabel(r'$k_x$ ($a^{-1}$)', fontsize = 18)
        ax.set_ylabel(r'$E$ ($\hbar v_F/a$)', fontsize = 18)
        ax.set_xlim(kmin,kmax)
        ax.set_ylim(-1,1)
        ax.axhline(0,ls = '--', c = 'gray', lw = 1)
    
        return bands, fig
    else:
        return bands
