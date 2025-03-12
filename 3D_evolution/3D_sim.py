from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch

import sys
# sys.path.append('../../')
import tensnet as tt

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda")

norm_preservation = False
bd_max = 150
compression_method = "randomized_svd"
encoding = "stacked"
N = 10

encoding_prefix = encoding[:3] #For saving and printing


# dt = 0.0002
# dt = 0.00002
dt = 0.00004 # 5 time steps correspond to 1 time step with dt = 0.0002

nu = 0.000665449275739045
Li = 10


ux_val = 1.0
Lx = Li
Ly = Li
Lz = Li
#  Setting various parameters
d = 2  # Physical dimensions
ax = 0.0
ay = 0.0
az = 0.0
bx = 1.0
by = 1.0
bz = 1.0

rad = 0.15
Re = np.round(ux_val * 2 * rad / nu)
eps = 1e-16

dx = (bx - ax) / (d**Lx - 1)
dy = (by - ay) / (d**Ly - 1)
dz = (bz - az) / (d**Lz - 1)

opt_method = "dmrg2"
opt_iter = 4  # DMRG Iterations
# 
tt.set_norm_preservation(norm_preservation)
tt.set_bd_limit(bd_max)
tt.set_comp_method(compression_method)
tt.set_arch(encoding)
tt.set_accuracy(eps)

tt.set_param("ax", ax)
tt.set_param("bx", bx)
tt.set_param("Lx", Lx)
tt.set_param("ay", ay)
tt.set_param("by", by)
tt.set_param("Ly", Ly)
tt.set_param("az", az)
tt.set_param("bz", bz)
tt.set_param("Lz", Lz)
tt.set_param("ux_val", ux_val)
tt.set_param(
    "low_acc", True
)  # Finite difference scheme (low-acc if True = 3rd order central diff)

cfl_cond = ux_val * dt / dx
cfl_cond

# 
var = int(datetime.now().timestamp())
# 
params = {
    "UID": var,
    "BD": bd_max,
    "Comp method": compression_method,
    "Encoding": encoding,
    "Time steps": N,
    "Opt method": opt_method,
    "CFL": np.round(cfl_cond, 2),
    "dt": dt,
    "Iterations DMRG": opt_iter,
    "Re": Re,
    "Norm preservation": norm_preservation,
}

print("Parameters:" + f"{params}\n", flush=True)
tt.save_object(params, f"results/param_{var}.pfile")

#  Initial Conditions
# print("Default initialization of velocity fields")
# ux = torch.load('u_8.tens').type(torch.float64)
# uy = torch.load('v_8.tens').type(torch.float64)
# uz = torch.load('w_8.tens').type(torch.float64)

# ux = torch.load('../Snapshots/tur_u.tens').type(torch.float64)
# uy = torch.load('../Snapshots/tur_v.tens').type(torch.float64)
# uz = torch.load('../Snapshots/tur_w.tens').type(torch.float64)

# ux = tt.vector_to_mps(ux,d,Lx + Ly + Lz)
# uy = tt.vector_to_mps(uy,d,Lx + Ly + Lz)
# uz = tt.vector_to_mps(uz,d,Lx + Ly + Lz)

# ux = ux.compress(max_bd=bd_max)
# uy = uy.compress(max_bd=bd_max)
# uz = uz.compress(max_bd=bd_max)

ux = torch.load(f'../MPS/tur_u_mps_{encoding_prefix}_BD_{bd_max}_torch.mps')
uy = torch.load(f'../MPS/tur_v_mps_{encoding_prefix}_BD_{bd_max}_torch.mps')
uz = torch.load(f'../MPS/tur_w_mps_{encoding_prefix}_BD_{bd_max}_torch.mps')

# Identity initialization
# ux = tt.identity_mps(L=Lx + Ly + Lz) * ux_val
# uy = tt.identity_mps(L=Lx + Ly + Lz) * 0.0
# uz = tt.identity_mps(L=Lx + Ly + Lz) * 0.0

# tt.plot_ux_uy_uz_rect(ux,uy,uz,z_slice=0,L=[Lx,Ly,Lz])

# Time Stepping
solution = tt.euler_time_stepping_3D(
    ux=ux,
    uy=uy,
    uz=uz,
    nofq=[Lx, Ly, Lz],
    dt=dt,
    dx=dx,
    dy=dy,
    dz=dz,
    N=N,
    nu=nu,
    # mask=mask_mps,
    bdtrunc=tt.BD_LIMIT(),
    opt_method=opt_method,
    # opt_accuracy=opt_acc,
    opt_iter=opt_iter,
    opt_paths_iter=100,
    saveit=1,
    UID=var,
    # inv_mpo=inv_A2,
)


# bcv = ["periodic", "periodic","periodic", "periodic", "periodic","periodic"]
# Dx1 = tt.get_diff_mpo_3D(dim=3, order=1, axis=0, bc=bcv) * (1.0 / dx)
# Dy1 = tt.get_diff_mpo_3D(dim=3, order=1, axis=1, bc=bcv) * (1.0 / dy)
# Dz1 = tt.get_diff_mpo_3D(dim=3, order=1, axis=2, bc=bcv) * (1.0 / dz)


tt.save_object(solution, f"results/{encoding_prefix}_{opt_method}_noq_{Lx+Ly+Lz}_BD_{bd_max}_iter_{N}_{var}.mpssol")

# t = 5
# tt.plot_ux_uy_uz_rect(solution[0][t], solution[1][t], solution[2][t], z_slice=0, L=[Lx,Ly,Lz])
# div_t = tt.get_3d_divergence(solution[0][t],solution[1][t],solution[2][t],Dx1,Dy1,Dz1,[Lx,Ly,Lz],bcv)
# plt.plot(div_t.contract())
