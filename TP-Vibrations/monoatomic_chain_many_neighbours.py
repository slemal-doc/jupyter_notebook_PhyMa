import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.cbook import flatten
from numpy import linalg as LA
from pylab import *

# set parameters

print(sys.argv)
for i in range(len(sys.argv)):
    if i > 0:
        sys.argv[i] = float(sys.argv[i])


#a = 2. # cell parameter, in angstrom
#M = [4.] # set M1 and M2

#C1 = [5.,0.,0.]  # preset forces constants, in order C1, C2, C3, ...
#C2 = [5.,4.,0.]

a = sys.argv[1] # cell parameter, in angstrom
M = [sys.argv[2]] # set M1 and M2

C1= [sys.argv[3], sys.argv[4], sys.argv[5]]  # preset forces constants, in order C1, C2, C3, ...
C2= [sys.argv[6], sys.argv[7], sys.argv[8]]


C = C1 # set force constants to the C1 configuration

number_of_atoms = len(M) # number of atoms
N =  len(C) # number of terms = number of force constants

# Next we define the dynamic matrix accounting for all set interactions. The returned value is the squared frequency.

def omega2(k):
    ssum = (1/M[0])*(4*C[0]*(np.sin(k*a/2)**2))
    sqrtsum = 0.
    for i in range(N):
        if i >0:
            ssum = ssum+(1/M[0])*(4*C[i]*(np.sin((i+1)*k*a/2)**2))
    return (ssum) # return the square of frequency at a given k-point 

# Number of k-points
nk = 5000

# Building a mesh in k-space to solve the dynamical matrix
kmesh = np.linspace(-np.pi/a,np.pi/a,nk)
            
# Plotting the frequencies
plt.figure(figsize=(12, 4), dpi=300) #
plt.subplot(1,3,1,adjustable='datalim')

plt.plot(kmesh, omega2(kmesh),'r', label='$C_1$=' + str(C[0]) + ',$C_2$=' + str(C[1]) + ',$C_3$=' + str(C[2]))
w0 = omega2(kmesh)
C = C2
wf = omega2(kmesh)
plt.plot(kmesh, omega2(kmesh),'b', label='$C_1$=' + str(C[0]) + ',$C_2$=' + str(C[1]) + ',$C_3$=' + str(C[2]))
plt.plot(kmesh, abs(wf-w0),'k--', label='Abs. difference')

plt.xlabel(r'k $(ang.^{-1})$')
plt.xlim(-np.pi/a,np.pi/a)
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a], ['$-\pi/a$','$-\pi/2a$','$0$','$\pi/2a$', '$\pi/a$']) # put tickmarks and labels at node positions

plt.ylabel(r'$\omega^2 (k)$')
#plt.ylim(0,)

plt.legend(loc='best')
plt.savefig("1D_atomic_chain_with_interaction_beyond_NN_1.pdf")

plt.subplot(1,3,2,adjustable='datalim')
plt.ylabel(r'$\omega (k)$')

C = C1
plt.plot(kmesh, np.sqrt(omega2(kmesh)),'r', label='$C_1$=' + str(C[0]) + ',$C_2$=' + str(C[1]) + ',$C_3$=' + str(C[2]))
w0 = omega2(kmesh)
C = C2 #
wf = omega2(kmesh)

plt.plot(kmesh, np.sqrt(omega2(kmesh)),'b', label='$C_1$=' + str(C[0]) + ',$C_2$=' + str(C[1]) + ',$C_3$=' + str(C[2]))

plt.xlabel(r'k $(ang.^{-1})$')
plt.xlim(-np.pi/a,np.pi/a)
plt.xticks([-np.pi/a,-np.pi/(2*a),0,np.pi/(2*a),np.pi/a], ['$-\pi/a$','$-\pi/2a$','$0$','$\pi/2a$', '$\pi/a$'])
plt.ylim(0,)

# Calculating the density of state (DOS) !

print('Calculating the DOS...')

ibz =   np.linspace(-np.pi/a,np.pi/a,nk) # 
solution_ibz = np.sqrt(omega2(ibz)) # We restrict the solution to the Brillouin zone
solution_list = list(flatten(solution_ibz)) # The solution is flattened

# For each frequencies w, we integrate the number of state between w-dw and w+dw
dw = 0.01
nw = int(max(solution_list)/dw)
w_space = np.linspace(0,max(solution_list),nw)
w_dos = []
for w in w_space:
    wsum = 0.
    for i in solution_list:
        if i > w-dw and i < w+dw:
            wsum = wsum + 1.
    w_dos.append(wsum)      

# Normalize the DOS so that the integral over the Brillouin zone is N !
dsum = 0.
dsum = sum(w_dos)
w_dos = w_dos/(dsum/number_of_atoms)
dsum = sum(w_dos)
    
print('Integrated DOS = ' + '%.2f'%(dsum))

plt.subplot(1,3,3,adjustable='datalim')
plt.plot(w_dos, w_space,'b', label='$C_1$=' + str(C[0]) + ',$C_2$=' + str(C[1]) + ',$C_3$=' + str(C[2]))
plt.legend(loc='best')
plt.xlabel(r'DOS')
plt.xlim(0,)
plt.ylim(0,)
plt.ylabel(r'$\omega$')
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.savefig("1D_atomic_chain_with_interaction_beyond_NN_2.pdf")


print ('')
