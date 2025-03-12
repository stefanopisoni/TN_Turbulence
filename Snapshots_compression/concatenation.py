# %% [markdown]
import sys
print(sys.path)
# %%
import numpy as np
sys.path.append('../')
import torch
from tensnet import *
import os
import h5py as h5

set_comp_method("svd")

# Upload the three components of a snapshot in MPS
u = torch.load('../MPS/tur_u_mps_sta_BD_2000_torch.mps')
v = torch.load('../MPS/tur_v_mps_sta_BD_2000_torch.mps')
w = torch.load('../MPS/tur_w_mps_sta_BD_2000_torch.mps')
print('loaded u,v,w')

# Create the new MPS with an extra tensor
t1 = torch.zeros((1, 3, 1))
t1[:, 0, :] = 1
t2 = torch.zeros((1, 3, 1))
t2[:, 1, :] = 1
t3 = torch.zeros((1, 3, 1))
t3[:, 2, :] = 1
q1 = MPS_Node(t1)
q2 = MPS_Node(t2)
q3 = MPS_Node(t3)
print('Additional qubits created')

new = MPS([q1] + u.nodes) + MPS([q2] + v.nodes) + MPS([q3] + w.nodes)
print('New bigger MPS created, with norm:')
print(new.norm)
print(new)


# Compress to different BD the new MPS and look at the l-2 norm differences
for BD_MAX in [100, 200, 500, 1000, 2000]:
    new1 = new.copy()
    new1 = new1.compress(max_bd = BD_MAX)
    print(f'Concatenated MPS with BD={BD_MAX} created. l_2 distance to original:')
    print((new + (new1 * -1.0)).norm)
    print("norm of the copmressed one:", new1.norm)

    # 3 single components
    u_new1 = new1.copy()
    v_new1 = new1.copy()
    w_new1 = new1.copy()
    u_new1[0] = u_new1[0].__getitem__(0)
    v_new1[0] = v_new1[0].__getitem__(1)
    w_new1[0] = w_new1[0].__getitem__(2)
    
    u_new1[1] = u_new1[0].contract(u_new1[1],["l f e, e q r -> l q r","... -> ..."])
    v_new1[1] = v_new1[0].contract(v_new1[1],["l f e, e q r -> l q r","... -> ..."])
    w_new1[1] = w_new1[0].contract(w_new1[1],["l f e, e q r -> l q r","... -> ..."])

    u_new1 = MPS(u_new1.nodes[1:31])
    v_new1 = MPS(v_new1.nodes[1:31])
    w_new1 = MPS(w_new1.nodes[1:31])
    print('MPS with fixed first tensor obtained')

    # Look at the difference btw u and u_new1
    print("l2 distance between the single components after compression of the concatenated MPS")
    print("l2(u,u_new1):",(u + (u_new1 * -1.0)).norm)
    print("l2(v,v_new1):",(v + (v_new1 * -1.0)).norm)
    print("l2(w,w_new1):",(w + (w_new1 * -1.0)).norm)

    # Get back the vectors
    snap_u = u_new1.contract()
    snap_u = snap_u.reshape((1024,1024,1024)).T
    snap_v = v_new1.contract()
    snap_v = snap_v.reshape((1024,1024,1024)).T
    snap_w = w_new1.contract()
    snap_w = snap_w.reshape((1024,1024,1024)).T
    print("Truncated snapshot recovered.")

    # Gradients
    u_grad_x = torch.gradient(snap_u, dim=0)
    v_grad_y = torch.gradient(snap_v, dim=1)
    w_grad_z = torch.gradient(snap_w, dim=2)

    # Divergence
    div = u_grad_x[0] + v_grad_y[0] + w_grad_z[0]
    print(f'divergence of concatenation with BD={BD_MAX} computed')

    # l_2 and l_inf of div
    l_2 = torch.linalg.vector_norm(div)
    l_inf = torch.linalg.norm(div.flatten(),ord=float('inf'))
    print('l_2:', l_2)
    print('l_inf:', l_inf)

    # Compute E(k)
    N = 10
    skip=2
    dim=int(round(2**N/skip))
    L=2*torch.pi

    uu_fft=torch.fft.fftn(snap_u[::skip,::skip,::skip])
    vv_fft=torch.fft.fftn(snap_v[::skip,::skip,::skip])
    ww_fft=torch.fft.fftn(snap_w[::skip,::skip,::skip])

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
    print('Spectrum obtained.')

    torch.save(spectrum, f'E_k_concatenation_{BD_MAX}.tens')