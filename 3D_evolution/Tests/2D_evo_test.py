# %%
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch

import sys
sys.path.append('../../')
import tensnet as tt

torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda")

# %%
ux_val = 1.0
N = 5
nu = 0.001
dt = 0.01
Lx = 6
Ly = 6
# %% Setting various parameters
d = 2  # Physical dimensions
ax = 0.0
ay = 0.0
bx = 1.0
by = 1.0

rad = 0.15
Re = np.round(ux_val * 2 * rad / nu)
eps = 1e-16

dx = (bx - ax) / (d**Lx - 1)
dy = (by - ay) / (d**Ly - 1)

opt_iter = 4  # DMRG Iterations
# %%
tt.set_bd_limit(150)
tt.set_accuracy(eps)
tt.set_arch("stacked")
tt.set_comp_method("svd")
tt.set_param("ax", ax)
tt.set_param("bx", bx)
tt.set_param("Lx", Lx)
tt.set_param("ay", ay)
tt.set_param("by", by)
tt.set_param("Ly", Ly)
tt.set_param("ux_val", ux_val)
tt.set_param(
    "low_acc", False
)  # Finite difference scheme (low-acc if True = 3rd order central diff)

cfl_cond = ux_val * dt / dx
cfl_cond

# %% 
var = int(datetime.now().timestamp())
# %%
params = {
    "lenx": bx - ax,
    "leny": by - ay,
    "Re": Re,
    "CFL Number": np.round(cfl_cond, 2),
    "NoQ": [Lx, Ly],
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
# ux = tt.random_mps(L=Lx + Ly, bd = 5, pd = d)
# uy = tt.random_mps(L=Lx + Ly, bd = 5, pd = d)
ux = tt.identity_mps(L=Lx + Ly) * ux_val
uy = tt.identity_mps(L=Lx + Ly) * 0.0

tt.plot_ux_uy_rect(ux, uy, L=[Lx,Ly])

# %% Time Stepping
solution = tt.euler_time_stepping(
    ux=ux,
    uy=uy,
    nofq=[Lx, Ly],
    dt=dt,
    dx=dx,
    dy=dy,
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
# tt.save_object(solution, f"../results/Re_{int(Re)}_{var}.mpssol")
tt.plot_ux_uy_rect(solution[0][-1], solution[1][-1], L=[Lx,Ly])

# %%
bcv = ["periodic", "periodic", "periodic", "periodic"]
Dx1 = tt.get_diff_mpo(dim=2, order=1, axis=0, bc=bcv) * (1.0 / dx)
Dy1 = tt.get_diff_mpo(dim=2, order=1, axis=1, bc=bcv) * (1.0 / dy)
div = tt.get_2d_divergence(solution[0][-1],solution[1][-1],Dx1,Dy1,[Lx,Ly],bcv)
plt.plot(div.contract())

# %%