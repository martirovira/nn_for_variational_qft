# IMPORTS ------------------------------------------------
import torch, time
from torch import nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt # Plotting library
from tqdm import tqdm # Progress bar
import ast
import copy
from matplotlib.ticker import StrMethodFormatter
import scipy as scp
from scipy.optimize import curve_fit
import os

# double precision tensors
torch.set_default_dtype(torch.float64)

# configuration functions

def array_to_lattice(psi_array, L):
    if L**3 != len(psi_array):
        print('L\u00b3 not equal to the number of inputs')
        return
    else:   
        psi_array_tensor = torch.tensor(psi_array)
        psi_tensor = torch.zeros(size=(L, L, L)) # kji = zyx

        # kji = zyx
        for k in range(L): 
            for j in range(L):
                psi_tensor[k,j,:] = psi_array_tensor[(L*k+j)*L:(L*k+j+1)*L]
        return psi_tensor
    
def lattice_to_array(psi_tensor, L):
    if L**3 != psi_tensor.numel():
        print('L\u00b3 not equal to the number of inputs')
        return
    else:
        psi_array_tensor = torch.zeros(L**3)
        for k in range(L): 
            for j in range(L):
                psi_array_tensor[(L*k+j)*L:(L*k+j+1)*L] = psi_tensor[k,j,:] 
                
        return psi_array_tensor
                
# HYPER-PARAMETERS ------------------------------------------------

# Harmonic Oscillator parameters
mu = 1
l0 = 0.15

# Lattice parameters
a = 1
L = 4
Nx = L
Ny = L
Nz = L

# NN hyperparameters
Nin = Nx * Ny * Nz # inputs to the neural network
Nout = 2  # outputs of the neural network
Nhid = 2*L**3 # number of nodes int the hidden layer

# Metropolis-hastings parameters
sigma = 1 # width of the proposal distribution
N_up = 8 # nmber of proposal sites to update at each MC step
N_therm = 3000 # configurations disregarded tu to thermalization
N_save = 400 # configuration generated for obtaining a decorrelated sample
x_0_tensor = torch.tensor([3.]*L**3).reshape(L,L,L) # initialization
x_0_array = lattice_to_array(x_0_tensor,L) 

# Gradient Descent and MonteCarlo parameters
epochs = 25 # gradient descent iterations
lr = 1 # learning rate
N_cf = int(1e4) # number of decorrelated configurations desired to generate

# OPEN THE FILE  ------------------------------------------------
file_path = 'nn_3d_training_'+str(L)+'x'+str(L)+'x'+str(L)+'/N_cf_'+str(N_cf)+'/lambda_'+str(l0)+'_N_up_'+str(int(N_up))+'_N_save_'+str(int(N_save))+'_N_therm_'+str(int(N_therm))+'_epochs_'+str(int(epochs))+'_N_hid_'+str(Nhid)+'.txt'
os.makedirs(os.path.dirname(file_path), exist_ok=True) # creates the file
file = open(file_path, 'w') # opens the file 

# K matrix ------------------------------------------------

def nabla_1d(mu,L):
    nabla = np.zeros(shape=(L,L))
    for a in range(L):
        for b in range(L):
            #print(a,b,(b+2)%L,(b-2)%L)
            if a == b:
                nabla[a][b] += 1/2
            if a == (b+2)%L:
                nabla[a][b] -= 1/4
            if a == (b-2)%L:
                nabla[a][b] -= 1/4
            #else:
             #   nabla[a][b] += 0
    return nabla 

def k(mu,L):
    nabla = nabla_1d(mu, L)
    I_1 = np.eye(L)
    a = np.kron(np.kron(nabla, I_1),I_1)
    b = np.kron(np.kron(I_1, nabla),I_1)
    c = np.kron(np.kron(I_1, I_1),nabla)
    nabla_3d = a+b+c
    return  mu**2 * np.eye(L**3,L**3) + nabla_3d

K = torch.real(torch.tensor(scp.linalg.sqrtm(k(1,4))))
#E = 0.5 * np.trace(K)
#print(0.5 * np.trace(K))

# NN ------------------------------------------------

# Hardware (CPU or GPU)
dev = 'cpu' # can be changed to 'cuda' for GPU usage
device = torch.device(dev)

# Network parameters.
seed = 1                                  # Seed of the random number generator
torch.manual_seed(seed)
W1 = torch.rand(Nhid, Nin, requires_grad=True) * (-1.) # First set of coefficients
B = torch.rand(Nhid, requires_grad=True) * 2. - 1.    # Set of bias parameters
W2 = torch.rand(Nout, Nhid, requires_grad=True)        # Second set of coefficients

class HarmonicNQS(nn.Module):
    def __init__(self, W1, B, W2):
        super(HarmonicNQS, self).__init__()
        
        # We set the operators 
        self.lc1 = nn.Linear(in_features=Nin, 
                             out_features=Nhid, 
                             bias=True)   # shape = (Nhid, Nin)
        self.actfun = nn.Sigmoid() # activation function
        self.lc2 = nn.Linear(in_features=Nhid, 
                             out_features=Nout, 
                             bias=False)  # shape = (Nout, Nhid)
        
        # We set the parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
   
    # We set the architecture
    def forward(self, x): 
        o = self.lc2(self.actfun(self.lc1(x)))
        exponent = - 0.5 * torch.matmul(x, torch.matmul(K, x))
        #psi = torch.exp(exponent) + 0*o 
        #return torch.tensor([1.,torch.dot(x,x)-torch.dot(x,x)], requires_grad=True)*psi
        return o * torch.exp(exponent)

net = HarmonicNQS(W1, B, W2).to(device)

# SUBROUTINES ------------------------------------------------

# Define the target distribution, which is the square of the ansatz (NN)
def target_distribution(x):
    x_tensor = torch.tensor(x.astype(float))
    psi = torch.view_as_complex(net(x_tensor)).reshape(1)
    p = torch.pow(torch.abs(psi),2)
    return  p.detach().numpy()[0]

def do_markov(target_distr, x_0, sigma, n, N_up):
    x = np.zeros(shape=(Nin, n))
    x[:,0] = x_0
    
    for i in range(1, n):
        current_x = x[:,i-1].copy()
        
        # Choose which components to update
        xi_to_update = np.random.choice(range(Nin), N_up, replace=False)
        
        # Generate a proposal
        proposed_x = current_x.copy()
        noise = np.random.multivariate_normal(np.zeros(N_up), sigma * np.identity(N_up))
        proposed_x[xi_to_update] += noise
        
        # Calculate acceptance ratio
        A = target_distr(proposed_x) / target_distr(current_x)

        if np.random.rand() <= A:
            x[:, i] = proposed_x  # accept move with probability min(1, A)
        else:
            x[:, i] = current_x  # otherwise "reject" move, and stay where we are
    return x

def metropolis_hastings(target_distr, x_0, sigma, N_cf, N_up, N_save):
    x = np.zeros(shape=(Nin, N_cf * N_save))
    x_finals = np.zeros(shape=(Nin, N_cf))
    
    if N_cf % N_save != 0:
        print('Error')
        
    x[:,0] = x_0
    x_finals[:,0] = x_0
    acceptance_ratio = 0
    
    j = 0
    
    for i in range(1, N_cf * N_save):
        current_x = x[:,i-1].copy()
        
        # Choose which components to update
        xi_to_update = np.random.choice(range(Nin), N_up, replace=False)
        
        # Generate a proposal
        proposed_x = current_x.copy()
        noise = np.random.multivariate_normal(np.zeros(N_up), sigma * np.identity(N_up))
        proposed_x[xi_to_update] += noise
        
        # Calculate acceptance ratio
        A = target_distr(proposed_x) / target_distr(current_x)

        if np.random.rand() <= A:
            x[:, i] = proposed_x  # accept move with probability min(1, A)
            acceptance_ratio += 1
        else:
            x[:, i] = current_x  # otherwise "reject" move, and stay where we are
            
        if i % N_save == 0:
            x_finals[:,j] = x[:,i]
            j += 1
            
    acceptance_ratio /= N_cf
    print('Acceptance Ratio:',acceptance_ratio*100,'%')
    return x_finals

def MCSampling(y,n): 
    integral =  (y.sum())/n #len(y)=len(x)=N
    error = np.std(y) / np.sqrt(n)
    return integral, error

def gradient_descent(D, integrand, model_weights):
    terms = (2 / N_cf) * np.real(D * integrand) 
    gradient = terms.sum(axis=1)

    new_weights = model_weights.copy() # what they will be the new 'w' & 'b'

    new_weights['lc1.weight'] -= gradient[0:Nhid*Nin].reshape(Nhid,Nin)
    new_weights['lc1.bias'] -= gradient[Nhid*Nin:Nhid*Nin+Nhid]
    new_weights['lc2.weight'] -= gradient[Nhid*Nin+Nhid:].reshape(Nout, Nhid)
    
    return new_weights

# TRAINING ------------------------------------------------
def training(epochs, N_cf):
    
    energies = np.zeros(epochs)
    energies_errors = np.zeros(epochs)
    IN = np.zeros(shape=(epochs, N_cf), dtype = complex)
    WF = np.zeros(shape=(epochs, N_cf), dtype = complex)

    
    for k in range(epochs):
        start_time = time.time()
        
        # Metropolis - Hastings
        if k == 0:
            # for the first epoch, we thermalise the initial x_0 config
            x_therm = do_markov(target_distribution, x_0_array, sigma, N_therm, N_up)
        else:
            # fo the rest of the epochs, we take the thermalised config as the last N_cf in the previous epoch
            x_therm = x_mh #the previous x_mh !!!!
        
        x_mh = metropolis_hastings(target_distribution, x_therm[:,-1], sigma, N_cf, N_up, N_save)

        # Compute ALL the derivatives 
        x = x_mh
        y = np.zeros(N_cf, dtype=complex)
        dy1 = np.zeros((Nin, N_cf), dtype=complex)
        dy2 = np.zeros((Nin, N_cf), dtype=complex)
        
        y2 = np.zeros(N_cf, dtype=complex)
        dyw1 = np.zeros(shape=(Nin, Nhid, N_cf), dtype=complex)
        dyb1 = np.zeros(shape= (Nhid, N_cf), dtype=complex)
        dyw2 = np.zeros(shape=(Nout, Nhid, N_cf), dtype=complex)
         
        differences = np.zeros(N_cf, dtype=complex)
        
        for i in range(N_cf):
            
            xi = torch.tensor(x[:,i].astype(float), requires_grad = True)
 
            # with respect to 'x'
            nni = net(xi)
            yi = torch.view_as_complex(nni).reshape(1)
            dy1ri = torch.autograd.grad(nni[0], xi, create_graph=True)[0]
            dy1ii = torch.autograd.grad(nni[1], xi, create_graph=True)[0]
            dy1i = dy1ri + dy1ii * 1j
            
            # Manually compute dy2/dx
            dy2ri = torch.zeros_like(xi)
            dy2ii = torch.zeros_like(xi)
            for j in range(len(xi)):
                dy2ri[j] = torch.autograd.grad(dy1ri[j], xi, create_graph=True)[0][j]
                dy2ii[j] = torch.autograd.grad(dy1ii[j], xi, create_graph=True)[0][j]
            dy2i = dy2ri + dy2ii * 1j

            y[i] = yi.detach().numpy()[0]
            dy1[:,i] = dy1i.detach().numpy()[0]
            dy2[:,i] = dy2i.detach().numpy()[0]
            net.zero_grad()
            
            # with respect to 'w', 'b'
            yi2 = torch.log(torch.conj(torch.view_as_complex(nni))).reshape(1)  #/ np.sqrt(N)
            yi2r = yi2.real
            yi2i = yi2.imag

            # Backpropagation for the real part
            yi2r.backward(retain_graph=True)
            dyw1ir = net.lc1.weight.grad.reshape(Nin, Nhid)
            dyb1ir = net.lc1.bias.grad
            dyw2ir = net.lc2.weight.grad

            net.zero_grad()  # Clear gradients for next part
            
            # Backpropagation for the imaginary part
            yi2i.backward(retain_graph=True)
            dyw1ii = net.lc1.weight.grad.reshape(Nin, Nhid)
            dyb1ii = net.lc1.bias.grad
            dyw2ii = net.lc2.weight.grad

            # Combine gradients (taking care of the imaginary unit)
            dyw1i = dyw1ir + 1j * dyw1ii
            dyb1i = dyb1ir + 1j * dyb1ii
            dyw2i = dyw2ir + 1j * dyw2ii

            y2[i] = yi2.detach().numpy()[0]
            dyw1[:,:,i] = dyw1i.detach().numpy()[0]
            dyb1[:,i] = dyb1i.detach().numpy()[0]
            dyw2[:,:,i] = dyw2i.detach().numpy()[0]
            
            # Reset gradients for next iteration
            net.zero_grad()
         
            # Manually compute finite differences
            
            x_tensor = array_to_lattice(x_mh[:,i],L)

            diff_lattice = 0

            for nz in range(L):
                for ny in range(L):
                    for nx in range(L):
                        diff_site = 0
                        
                        # PBC for x-axis
                        if nx == 0:
                            diff_site += (x_tensor[nz,ny,nx+1]-x_tensor[nz,ny,L-1])**2 # i=1,3
                        elif nx == L-1:
                            diff_site += (x_tensor[nz,ny,0]-x_tensor[nz,ny,nx-1])**2 #i=0,2
                        else:
                            diff_site += (x_tensor[nz,ny,nx+1]-x_tensor[nz,ny,nx-1])**2 
                            
                        # PBC for y-axis
                        if ny == 0:
                            diff_site += (x_tensor[nz,ny+1,nx]-x_tensor[nz,L-1,nx])**2 # j=1,3
                        elif ny == L-1:
                            diff_site += (x_tensor[nz,0,nx]-x_tensor[nz,ny-1,nx])**2 #j=0,2
                        else:
                            diff_site += (x_tensor[nz,ny+1,nx]-x_tensor[nz,ny-1,nx])**2
                        
                        # PBC for z-axis
                        if nz == 0:
                            diff_site += (x_tensor[nz+1,ny,nx]-x_tensor[L-1,ny,nx])**2 # k=1,3
                        elif nz == L-1:
                            diff_site += (x_tensor[0,ny,nx]-x_tensor[nz-1,ny,nx])**2 # k=0,2
                        else:
                            diff_site += (x_tensor[nz+1,ny,nx]-x_tensor[nz-1,ny,nx])**2
                        
                        diff_lattice += diff_site

            differences[i] = diff_lattice
       
        # Monte Carlo
        t = -(1 / (2 * a**3)) * y**(-1) * dy2.sum(axis=0)
        u1 = (1/2) * (1 / (2*a)**2) * differences
        u2 = (1/2) * mu**2 * np.sum(x**2, axis=0)
        u3 = l0 * np.sum(x**4, axis=0)
        integrand = t + a**3 * (u1 + u2 + u3)
        IN[k] = integrand
        WF[k] = y

        energies[k], energies_errors[k] = MCSampling(np.real(integrand), N_cf)
        
        # SAVE THE DATA
        line = f"{k} {energies[k]} {energies_errors[k]}\n"
        file.write(line)
        # instantly writing
        file.flush()
        os.fsync(file.fileno())
        
        # Gradient descent 
        D = np.concatenate((dyw1.reshape(Nin*Nhid,N_cf),dyb1,dyw2.reshape(Nout*Nhid,N_cf)))
        model_weights = net.state_dict() # current 'w' & 'b' of the NN
        new_weights = gradient_descent(D, integrand, model_weights)
        
        ## update the weights and we go to the next epoch
        net.load_state_dict(new_weights)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
    return x_mh, WF, IN, energies, energies_errors


x_mh, WF, IN, energies, energies_errors = training(epochs, N_cf)
file.close() # closes the file

# PLOTTING ------------------------------------------------
# energy vs epoch
plt.errorbar(np.arange(1, epochs+ 1, 1), energies, energies_errors, label = 'MC energy', marker = 'o', markersize = 4, capsize = 5)
plt.plot([1, epochs+1], [49.75517243930367, 49.75517243930367], linestyle = 'dotted', color = 'black', label = 'Exact GS energy')
plt.xlabel('epoch')
plt.ylabel('E / ℏω')
plt.legend(loc='upper right')
plt.savefig('nn_3d_training_'+str(L)+'x'+str(L)+'x'+str(L)+'/N_cf_'+str(N_cf)+'/'+'lambda_'+str(l0)+'_N_up_'+str(int(N_up))+'_N_save_'+str(int(N_save))+'_N_therm_'+str(int(N_therm))+'_epochs_'+str(int(epochs))+'_N_hid_'+str(Nhid)+'.pdf',bbox_inches='tight')

# FINAL ENERGY ------------------------------------------------
E_star = energies[-1]
dE_star = energies_errors[-1]
print(f'Variational estimate of GS energy for λ = {l0: .4f} : ({E_star: .3f} \u00B1{dE_star: .3f} ) ħω' )
