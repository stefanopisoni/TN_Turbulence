# %%
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch

import sys
sys.path.append('../../../tensnet/src/')
import tensnet as tt # type: ignore

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda")


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

# %%
norm_preservation = True #Compress first if using this option
bd_max = 20
dt = 0.0002
N = 5
nu = 0.00066
Li = 6
# nu = 0.000665449275739045

ux_val = 1.0
Lx = Li
Ly = Li
Lz = Li
# %% Setting various parameters
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

opt_iter = 4  # DMRG Iterations
# %%
tt.set_norm_preservation(norm_preservation)
tt.set_bd_limit(bd_max)
tt.set_accuracy(eps)
tt.set_arch("stacked")
tt.set_comp_method("randomized_svd")
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

# %% 
var = int(datetime.now().timestamp())
# %%
params = {
    "norm_preservation": norm_preservation,
    "lenx": bx - ax,
    "leny": by - ay,
    "lenz": bz - az,
    "Re": Re,
    "CFL Number": np.round(cfl_cond, 2),
    "NoQ": [Lx, Ly, Lz],
    "dx": dx,
    "dt": dt,
    "low_acc": tt.parameters("low_acc"),
    # "mask": mask_mps,
    "ux_val": ux_val,
    "visc": nu,
    "Iteration limit for DMRG": opt_iter,
    "TimeSteps": N,
    "UID": var,
    "config_params": tt.parameters(),
}

print("Parameters:" + f"{params}")
# tt.save_object(params, f"../results/param_{var}.pfile")

# %% Initial Conditions
print("Default initialization of velocity fields")
ux = torch.load('/Users/stefanopisoni/NoSync/3D_solver/u_6.tens').type(torch.float64)
uy = torch.load('/Users/stefanopisoni/NoSync/3D_solver/v_6.tens').type(torch.float64)
uz = torch.load('/Users/stefanopisoni/NoSync/3D_solver/w_6.tens').type(torch.float64)

fig = plt.figure(figsize=(3,3))
a = fig.add_subplot(111)
p = a.imshow(ux[:,0,:])

# UNcomment for sequential encoding
# ux = shuffle_axis_3D(ux.T,6)
# uy = shuffle_axis_3D(uy.T,6)
# uz = shuffle_axis_3D(uz.T,6)


ux = tt.vector_to_mps(ux,d,Lx + Ly + Lz)
uy = tt.vector_to_mps(uy,d,Lx + Ly + Lz)
uz = tt.vector_to_mps(uz,d,Lx + Ly + Lz)

ux = ux.compress(max_bd=bd_max)
uy = uy.compress(max_bd=bd_max)
uz = uz.compress(max_bd=bd_max)

# tt.plot_ux_uy_uz_rect(ux,uy,uz,z_slice=0,L=[Lx,Ly,Lz])

# PLOT sequential encoding
# snap = ux.contract()
# snap = unshuffle_axis_3D(snap.reshape((d**Li,d**Li,d**Li)),Li).T
# fig = plt.figure(figsize=(3,3))
# a = fig.add_subplot(111)
# p = a.imshow(snap[:,0,:])

# %% Time Stepping
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
    opt_method="dmrg2",
    # opt_accuracy=opt_acc,
    opt_iter=opt_iter,
    opt_paths_iter=100,
    saveit=100,
    UID=var,
    # inv_mpo=inv_A2,
)


# %%
bcv = ["periodic", "periodic","periodic", "periodic", "periodic","periodic"]
Dx1 = tt.get_diff_mpo_3D(dim=3, order=1, axis=0, bc=bcv) * (1.0 / dx)
Dy1 = tt.get_diff_mpo_3D(dim=3, order=1, axis=1, bc=bcv) * (1.0 / dy)
Dz1 = tt.get_diff_mpo_3D(dim=3, order=1, axis=2, bc=bcv) * (1.0 / dz)



# %%
# tt.save_object(solution, f"results/noq_{Lx+Ly+Lz}_BD_{bd_max}_iter_{N}.mpssol")

# %%
t = 0
tt.plot_ux_uy_uz_rect(solution[0][t], solution[1][t], solution[2][t], z_slice=0, L=[Lx,Ly,Lz])
# div_t = tt.get_3d_divergence(ux,uy,uz,Dx1,Dy1,Dz1,[Lx,Ly,Lz],bcv)
div_t = tt.get_3d_divergence(solution[0][t],solution[1][t],solution[2][t],Dx1,Dy1,Dz1,[Lx,Ly,Lz],bcv)
plt.plot(div_t.contract())
# plt.xscale('log')
# %%
