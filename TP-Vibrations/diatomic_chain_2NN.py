import numpy as np
from scipy.integrate import odeint
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from numpy import linalg as LA
from pylab import *

########################################
#                                      #
#        Part 1: set parameters        #
#                                      #
########################################

M = [1.,1.] # set M1 and M2
C = [3.,3.] # set C1 and C2
K = [-1,6.] # set K1 and K2
a = 1. # cell parameter
N = len(M) 

print (str(N) + " atoms in the cell -> 2 bands along k." + '\n')

########################################
#                                      #
#      Part 2: functions returning     #
#   the dynamical matrix and the DOS   #
#                                      #
########################################

def dyn_mat_2at(k):
    return np.array([[(C[0]+C[1]+2*K[0]-K[0]*(np.exp(1.j*k*a) + np.exp(-1.j*k*a)))/M[0], (-1/np.sqrt(M[0]*M[1]))*(C[0] + C[1]*np.exp(-1.j*k*a))],\
                     [(-1/np.sqrt(M[0]*M[1]))*(C[0] + C[1]*np.exp(1.j*k*a)), (C[0]+C[1]+2*K[1]-K[1]*(np.exp(1.j*k*a) + np.exp(-1.j*k*a)))/M[1]]],\
                    dtype=complex)

def dos_w(solution,N):
    # For each frequencies w of a given solution (eigenvalues), 
    # we integrate the number of state between w-dw and w+dw
    # Only the solution within the Brillouin zone must be provided
    solution_list = list(flatten(solution))

    dw = 3e-3
    nw = int(max(solution_list)/dw)
    w_space = np.linspace(-0.1,max(solution_list)+0.1,nw)
    w_dos = []

    for w in w_space:
        wsum = 0.
        for i in solution_list:
            if i > w-dw and i < w+dw:
                wsum = wsum + 1
        w_dos.append(wsum)  
        
    # Normalize the DOS so that the integral over the Brillouin zone is N !
    w_sum = sum(w_dos)*dw
    w_dos = (w_dos/(w_sum/N))
    w_sum = sum(w_dos)*dw

    return w_space,w_dos,w_sum

print ('Dynamical matrix at Gamma (k = 0):')
print (dyn_mat_2at(0))
print ('\nThe corresponding eigenvalues w^2 are:')
print (LA.eigh(dyn_mat_2at(0))[0])
print ('\nThe corresponding eigenvectors are:')
print (LA.eigh(dyn_mat_2at(0))[1])
print ('\n')


########################################
#                                      #
#      Part 3: diagonalizing the       #
#   dynamical matrix on a mesh in      #
#    k-space and computing the DOS     #
#  using the eigenvalues restricted to #
#      the first Brillouin zone        #
#                                      #
########################################

nk = 400 # Number of k-points

kmesh = np.linspace(-1*np.pi,2*np.pi,nk) # Building a k-mesh in k-space to solve the dynamical matrix

# building the array where the solutions will be stocked
# Due to numerical accurary issues, some solutions at a given k-point will be "Not a number" (NaN).
# These are rejected, and we use the k-points without any pathologies for the mesh.

eigenvalues = []
eigenvectors = []
kokay = [] 
for i in range(N):
    eigenvalues.append([])
    eigenvectors.append([])
    kokay.append([])
    
#Diagonalization of the dynamical matrix for each k points. 
#The number of solutions is equal to N for a 1D system.
#We only take the positive roots.

for i in range(N):
    for k in kmesh:
        n = LA.eigh(dyn_mat_2at(k))
        if np.real(n[0][i]) > 0:  
            eigenvalues[i].append(np.sqrt(np.real(n[0][i])))
            eigenvectors[i].append((np.real((n[1]))))
            kokay[i].append(k)

# Calculation the DOS: the previous step are performed again, only within the Brillouin zone

print('Calculating the DOS...')

ibz = np.linspace(-np.pi/a,np.pi/a,nk)
large_ibz = np.linspace(-np.pi/a,np.pi/a,nk*40)
eigenvalues_ibz = [] 
kokay_ibz = []    # 
for i in range(N):
    eigenvalues_ibz.append([])
    kokay_ibz.append([])

for i in range(N):
    for k in ibz:
        n = LA.eigvalsh(dyn_mat_2at(k))
        if np.real(n[i]) > 0:  
            eigenvalues_ibz[i].append(np.sqrt(np.real(n[i])))
            kokay_ibz[i].append(k)

# We expand the solution to more k-points by interpolation
exp_eigenvalues_ibz = eigenvalues_ibz
for i in range(N):
    s = interpolate.InterpolatedUnivariateSpline(kokay_ibz[i], eigenvalues_ibz[i])
    exp_eigenvalues_ibz[i] = s(large_ibz)

max_w = max(list(flatten(eigenvalues_ibz)))
      
results_dos = [dos_w(exp_eigenvalues_ibz,N)[0],\
               dos_w(exp_eigenvalues_ibz,N)[1],\
               dos_w(exp_eigenvalues_ibz,N)[2]]

print('Integrated DOS = ' + '%.2f'%(results_dos[2]))

########################################
#                                      #
#     Part 4: Plotting the results     #
#                                      #
########################################

plt.figure(figsize=(12, 12), dpi=300) 
plt.subplot(2,2,1,adjustable='datalim')

plt.plot(kokay[0],eigenvalues[0],'ro',ms=1., label='band ' + str(1))
plt.plot(kokay[1],eigenvalues[1],'go',ms=1., label='band ' + str(2))
#plt.plot(kmesh,np.sqrt((C[0]+C[1])*(M[0]+M[1])/(M[0]*M[1]))*abs(np.sin(kmesh*a/4)),\
#         'b--',label='Monoatomic dispersion')
    
plt.xlabel(r'k')
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a, (1.5)*np.pi/a, 2*np.pi/a],\
           ['$-\pi/a$','$-\pi/2a$','$0$','$\pi/2a$', '$\pi/a$', '$3\pi/2a$', '$2\pi/a$'])
plt.xlim(-np.pi/a,2*np.pi/a)
plt.axvline(x=np.pi/a, c='gray')
plt.ylabel(r'$\omega (k)$')
plt.ylim(0,max_w+0.2)
plt.legend(loc='best')

plt.subplot(2,2,2,adjustable='datalim')
plt.xlabel(r'DOS')
plt.ylabel(r'$\omega$')
plt.ylim(0,max_w+0.2)
plt.xlim(0,14)
plt.plot(results_dos[1], results_dos[0], 'b', ms= 0.5)
plt.savefig("1D_diatomic_chain_vibrations_2NN.pdf")

print ('')
