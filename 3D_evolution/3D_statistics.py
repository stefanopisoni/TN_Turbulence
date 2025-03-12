from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch

import os

import sys
# sys.path.append('../../tensnet/src/')
import tensnet as tt

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda")



# XYZXYZ DECODING FUNCTION
def unshuffle_axis_3D(A: torch.Tensor, L: int):
    # tensorize A and unshuffle odd and even axis and return a matrix
    axis_even_odd = [x for x in range(3 * L) if x % 3 == 0] + [
        x for x in range(3 * L) if x % 3 == 1] + [
        x for x in range(3 * L) if x % 3 == 2
    ]
    A = A.reshape([2] * 3 * L).permute(axis_even_odd).reshape(2**L, 2**L, 2**L)
    return A

print("Unshuffling function defined.")


Li = 10
d = 2

encoding_prefix = "seq"
opt_method = "dmrg2"
noq = 3 * Li
bd_max = 100
T = 20

var = 1741605290


# Change BD accordingly here and below in the saved files
solution = tt.load_object(f'../3D_solver/results/{encoding_prefix}_{opt_method}_noq_{noq}_BD_{bd_max}_iter_{T}_{var}.mpssol')
print(f"{encoding_prefix}_{opt_method}_noq_{noq}_BD_{bd_max}_iter_{T}_{var} LOADED")


u = []
v = []
w = []

for i in range(T+1):
    u.append(solution[0][i])
    v.append(solution[1][i])
    w.append(solution[2][i])


# E(k) and div(u) for every time step.

for time in range(T+1):

    if encoding_prefix == "sta":

        u_ans = u[time].contract().reshape(1024,1024,1024)
        v_ans = v[time].contract().reshape(1024,1024,1024)
        w_ans = w[time].contract().reshape(1024,1024,1024)

    elif encoding_prefix == "seq":

        u_ans = u[time].contract()
        u_ans = unshuffle_axis_3D(u_ans.reshape((1024,1024,1024)),Li).T
        v_ans = v[time].contract()
        v_ans = unshuffle_axis_3D(v_ans.reshape((1024,1024,1024)),Li).T
        w_ans = w[time].contract()
        w_ans = unshuffle_axis_3D(w_ans.reshape((1024,1024,1024)),Li).T


    # Compute E(K) and div with this components
    N = Li
    skip=2
    dim=int(round(2**N/skip))
    L=2*torch.pi

    uu_fft=torch.fft.fftn(u_ans[::skip,::skip,::skip])
    vv_fft=torch.fft.fftn(v_ans[::skip,::skip,::skip])
    ww_fft=torch.fft.fftn(w_ans[::skip,::skip,::skip])

    uu_fft=(torch.abs(uu_fft)/dim**3)**2
    vv_fft=(torch.abs(vv_fft)/dim**3)**2
    ww_fft=(torch.abs(ww_fft)/dim**3)**2

    k_end=int(dim/2)
    rx=torch.Tensor(range(dim))-dim/2+1
    rx=torch.roll(rx,int(dim/2)+1)

    r=torch.zeros((rx.shape[0],rx.shape[0],rx.shape[0]))
    for i in range(rx.shape[0]):
        for j in range(rx.shape[0]):
                r[i,j,:]=rx[i]**2+rx[j]**2+rx[:]**2
    r=torch.sqrt(r)

    dx=2*torch.pi/L
    k=(torch.tensor(range(k_end))+1)*dx

    bins=torch.zeros((k.shape[0]+1))
    for N in range(k_end):
        if N==0:
            bins[N]=0
        else:
            bins[N]=(k[N]+k[N-1])/2    
    bins[-1]=k[-1]

    inds = torch.bucketize(r*dx, bins)
    spectrum=torch.zeros((k.shape[0]))
    bin_counter=torch.zeros((k.shape[0]))

    for N in range(k_end):
        spectrum[N]=torch.sum(uu_fft[inds==N+1])+torch.sum(vv_fft[inds==N+1])+torch.sum(ww_fft[inds==N+1])
        bin_counter[N]=torch.count_nonzero(inds==N+1)

    spectrum=spectrum*2*torch.pi*(k**2)/(bin_counter*dx**3)


    repo_name = f'stat_tt/{encoding_prefix}_{opt_method}_noq_{noq}_BD_{bd_max}_iter_{T}_{var}'
    os.makedirs(repo_name, exist_ok=True)


    # torch.save(spectrum, f'stat_tt/dmrg2_seq_E_k_tt_BD150_noq10_{time}.tens')
    # Create repo with the corresponding address
    torch.save(spectrum, f'{repo_name}/E_k_{time}.tens')

    # Divergence
    u_grad_x = torch.gradient(u_ans, dim=0)
    v_grad_y = torch.gradient(v_ans, dim=1)
    w_grad_z = torch.gradient(w_ans, dim=2)
    
    #divenrgence
    div = u_grad_x[0] + v_grad_y[0] + w_grad_z[0]
    # torch.save(div, f'div_tt_BD100_noq10_{time}.tens')

    #l_2 and l_inf of div
    l_2 = torch.linalg.vector_norm(div)
    l_inf = torch.linalg.norm(div.flatten(),ord=float('inf'))

    # torch.save(l_2, f'stat_tt_dmrg2/dmrg2_seq_div_tt_l2_BD150_noq10_{time}.tens')
    torch.save(l_2, f'{repo_name}/div_l2_{time}.tens')
    # torch.save(l_inf, f'stat_tt_dmrg2/dmrg2_seq_div_tt_linf_BD150_noq10_{time}.tens')
    torch.save(l_inf, f'{repo_name}/div_linf_{time}.tens')
    
    print(f'time step {time}')
    print(l_2)
    print(l_inf)