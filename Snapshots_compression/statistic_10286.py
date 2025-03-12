# %% [markdown]
import sys
print(sys.path)
# %%
import numpy as np
sys.path.append('../')
import torch
from tensnet import *
# torch.set_default_device('cuda')
import os
import h5py as h5

#XYZXYZ ENCODING FUNCTION
def shuffle_axis_3D(A: torch.Tensor, L: int):
    # tensorize A and shuffle odd and even axis and return a matrix
    new_axis = []
    for x in range(L):
        new_axis += [x, x + L, x + 2*L]
    A = A.reshape([2] * 3 * L).permute(new_axis).reshape(2**L, 2**L, 2**L)
    return A

# XYZXYZ DECODING FUNCTION
def unshuffle_axis_3D(A: torch.Tensor, L: int):
    # tensorize A and unshuffle odd and even axis and return a matrix
    axis_even_odd = [x for x in range(3 * L) if x % 3 == 0] + [
        x for x in range(3 * L) if x % 3 == 1] + [
        x for x in range(3 * L) if x % 3 == 2
    ]
    A = A.reshape([2] * 3 * L).permute(axis_even_odd).reshape(2**L, 2**L, 2**L)
    return A
print("Succesfully defined sequencial encoding functions.")


directory = '../../raw_data'
raw_data = 10286


filename = f'hit.1024.{raw_data}.h5'
f = os.path.join(directory, filename)
# checking if it is a file
if os.path.isfile(f):
    print(f)

N=1024 #Size of the box in points
Ny=N
Nx=N
Nz=N/2+1
filename =  f #File to read
ID = filename[14:-3]

print("Loading file")
print(ID)

f = h5.File(filename, 'r')

print("File load complete")

up= f['u'][:] #Read k_x=0 plane of u
v = f['v'][:] #Read v
w = f['w'][:] #Read w
up.dtype='c8'
v.dtype ='c8'
w.dtype ='c8'

u=np.zeros((N,N,N//2+1),dtype='c8')
u[0,:,:]=up    

time   = f['time'][:]
n_file = f['n_file'][:]
nu     = f['nu'][:]

k=np.r_[0:N//2,-N//2:0]
kx=k.reshape((-1, 1, 1))
ky=k.reshape(( 1,-1, 1))
kz=k[0:(N//2+1)].reshape(( 1, 1,-1))

u[1:,...]=((-ky*v-kz*w)/kx)[1:,...]

print("Starting fft")
u=np.fft.irfftn(u)
v=np.fft.irfftn(v)
w=np.fft.irfftn(w)
print("Finished fft")

print("Converting to torch")
u = torch.from_numpy(u)
v = torch.from_numpy(v)
w = torch.from_numpy(w)


# gradients
u_grad = torch.gradient(u)
# v_grad = torch.gradient(v)
# w_grad = torch.gradient(w)
print("gradient of u computed")

# Flattening to vectors to get the XXYYZZ-encoded MPS
psi_u = u.T
psi_u = psi_u.flatten()
psi_v = v.T
psi_v = psi_v.flatten()
psi_w = w.T
psi_w = psi_w.flatten()
print("Succesfully flattened data sta.")

#Reshuffling according to XYZXYZ
u_new = shuffle_axis_3D(u.T,10)
v_new = shuffle_axis_3D(v.T,10)
w_new = shuffle_axis_3D(w.T,10)
#Flattening data to vectors to get the XYZXYZ-encoded MPS
phi_u = u_new.flatten()
phi_v = v_new.flatten()
phi_w = w_new.flatten()
print("Succesfully flattened data seq.")


for BD_MAX in [100,200,500,1000,2000,5000]:

    encoding = 'sta'

    # Vector to MPS
    tur_u_mps = vector_to_mps(psi_u, 2, 30, None, BD_MAX)
    tur_v_mps = vector_to_mps(psi_v, 2, 30, None, BD_MAX)
    tur_w_mps = vector_to_mps(psi_w, 2, 30, None, BD_MAX)
    print("vector to MPS completed")
    print(tur_u_mps)

    # MPS to vect
    snap_u = tur_u_mps.contract()
    snap_u = snap_u.reshape((1024,1024,1024)).T
    snap_v = tur_v_mps.contract()
    snap_v = snap_v.reshape((1024,1024,1024)).T
    snap_w = tur_w_mps.contract()
    snap_w = snap_w.reshape((1024,1024,1024)).T
    print("Truncated snapshot recovered.")

    #l_2_vectors
    l_2_u = torch.linalg.vector_norm(u-snap_u)
    torch.save(l_2_u, f'l_2_{raw_data}_u_{encoding}_{BD_MAX}.tens')
    l_2_v = torch.linalg.vector_norm(v-snap_v)
    torch.save(l_2_v, f'l_2_{raw_data}_v_{encoding}_{BD_MAX}.tens')
    l_2_w = torch.linalg.vector_norm(w-snap_w)
    torch.save(l_2_w, f'l_2_{raw_data}_w_{encoding}_{BD_MAX}.tens')

    #gradients
    u_grad_BD = torch.gradient(snap_u)
    v_grad_BD = torch.gradient(snap_v, dim=1)
    w_grad_BD = torch.gradient(snap_w, dim=2)

    #l_2_grad
    l_2_grad = 0
    for i in [0,1,2]:
        l_2_grad = l_2_grad + torch.linalg.norm(u_grad[i]-u_grad_BD[i])
    l_2_grad = l_2_grad/3

    torch.save(l_2_grad, f'l_2_grad_{raw_data}_u_{encoding}_{BD_MAX}.tens')
    print(l_2_grad)

    #divenrgence
    div = u_grad_BD[0] + v_grad_BD[0] + w_grad_BD[0]

    #l_2 and l_inf of div
    l_2 = torch.linalg.vector_norm(div)
    l_inf = torch.linalg.norm(div.flatten(),ord=float('inf'))

    torch.save(l_2, f'div_l_2_norm_{raw_data}_{encoding}_{BD_MAX}.tens')
    torch.save(l_inf, f'div_l_inf_norm_{raw_data}_{encoding}_{BD_MAX}.tens')
    print(l_2)
    print(l_inf)


    encoding = 'seq'

    # Vector to MPS
    tur_u_mps = vector_to_mps(phi_u, 2, 30, None, BD_MAX)
    tur_v_mps = vector_to_mps(phi_v, 2, 30, None, BD_MAX)
    tur_w_mps = vector_to_mps(phi_w, 2, 30, None, BD_MAX)
    print("vector to MPS seq completed")
    print(tur_u_mps)

    # MPS to vect
    snap_u = tur_u_mps.contract()
    snap_u = unshuffle_axis_3D(snap_u.reshape((1024,1024,1024)),10).T
    snap_v = tur_v_mps.contract()
    snap_v = unshuffle_axis_3D(snap_v.reshape((1024,1024,1024)),10).T
    snap_w = tur_w_mps.contract()
    snap_w = unshuffle_axis_3D(snap_w.reshape((1024,1024,1024)),10).T
    print("Compressed snapshot recovered.")

    # l_2_vectors
    l_2_u = torch.linalg.vector_norm(u-snap_u)
    torch.save(l_2_u, f'l_2_{raw_data}_u_{encoding}_{BD_MAX}.tens')
    l_2_v = torch.linalg.vector_norm(v-snap_v)
    torch.save(l_2_v, f'l_2_{raw_data}_v_{encoding}_{BD_MAX}.tens')
    l_2_w = torch.linalg.vector_norm(w-snap_w)
    torch.save(l_2_w, f'l_2_{raw_data}_w_{encoding}_{BD_MAX}.tens')

    #gradients
    u_grad_BD = torch.gradient(snap_u)
    v_grad_BD = torch.gradient(snap_v, dim=1)
    w_grad_BD = torch.gradient(snap_w, dim=2)

    #l_2_grad
    l_2_grad = 0
    for i in [0,1,2]:
        l_2_grad = l_2_grad + torch.linalg.norm(u_grad[i]-u_grad_BD[i])
    l_2_grad = l_2_grad/3

    torch.save(l_2_grad, f'l_2_grad_{raw_data}_u_{encoding}_{BD_MAX}.tens')
    print(l_2_grad)
    
    #divergence
    div = u_grad_BD[0] + v_grad_BD[0] + w_grad_BD[0]

    #l_2 and l_inf of div
    l_2 = torch.linalg.vector_norm(div)
    l_inf = torch.linalg.norm(div.flatten(),ord=float('inf'))

    torch.save(l_2, f'div_l_2_norm_{raw_data}_{encoding}_{BD_MAX}.tens')
    torch.save(l_inf, f'div_l_inf_norm_{raw_data}_{encoding}_{BD_MAX}.tens')
    print(l_2)
    print(l_inf)