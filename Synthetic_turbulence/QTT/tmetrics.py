# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from tutils import *
from cascade import *

# %% Configuration
nscales  = 10
N        = 2**nscales
L        = 2 * np.pi
nrank    = 5
epsilon  = 1.0
var      = 1.0
eps      = 1e-8
max_sep  = 128
num_runs = 8
initial = 13

# %% Precompute radial bins
kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
ky = kx
kz = 2 * np.pi * np.fft.rfftfreq(N, d=L/N)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
KX_shifted = np.fft.fftshift(KX)
KY_shifted = np.fft.fftshift(KY)
KZ_shifted = np.fft.fftshift(KZ)
k_mag = np.sqrt(KX_shifted**2 + KY_shifted**2 + KZ_shifted**2)

k_min     = 1.0
k_max     = k_mag.max() / np.sqrt(3)
num_bins  = N // 2
k_bins    = np.linspace(k_min, k_max, num_bins + 1)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
k_magnitude_flat = k_mag.ravel()

# Define bins in k (spherical shells)

k_min = k_magnitude_flat[k_magnitude_flat > 0].min()  # exclude zero to avoid log(0)
k_max = k_magnitude_flat.max()/1.7
bins = np.linspace(0.0, k_max, num_bins+1)  # num_bins bins
bin_centers = 0.5 * (bins[:-1] + bins[1:])
delta_k = bins[1:] - bins[:-1]

# %% Main loop
flat_records   = []
energy_records = []

for run_idx in range(initial,initial+num_runs):
    print("run", run_idx)
    Ax, Ay, Az = gen_TT_cascade_3d(
        Nscales=nscales, nrank=nrank, levels=None, seed=None,
        epsilon=epsilon, method='skc', 
        eps=eps, var=var, boundary='periodic', order=2
    )
    tntt.save(Ax, f'./data/mps/10_0/Ax_run{run_idx}.TT')
    tntt.save(Ay, f'./data/mps/10_0/Ay_run{run_idx}.TT')
    tntt.save(Az, f'./data/mps/10_0/Az_run{run_idx}.TT')

    A_x = z_order_to_normal_nopara(Ax.full().cpu().numpy().ravel(), np.zeros((N, N, N)), N, N, N)
    A_y = z_order_to_normal_nopara(Ay.full().cpu().numpy().ravel(), np.zeros((N, N, N)), N, N, N)
    A_z = z_order_to_normal_nopara(Az.full().cpu().numpy().ravel(), np.zeros((N, N, N)), N, N, N)

    u, v, w = compute_velocity_from_vector_potential(A_x, A_y, A_z, L)

    u_hat = np.fft.rfftn(u)
    v_hat = np.fft.rfftn(v)
    w_hat = np.fft.rfftn(w)
    E_hat = (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2) / (N**6) /2
    E_hat_shifted = np.fft.fftshift(E_hat)

    E_flat = E_hat.ravel()
    
    # 3. Get wavenumbers in 3D.
    e_flat = E_hat_shifted.flatten()


    # Bin the energy using np.histogram with weights:
    bin_energy, _ = np.histogram(k_magnitude_flat, bins=bins, weights=e_flat)
    # Normalize by the bin width to obtain an estimate of the energy density E(k)
    E_k = bin_energy / delta_k

    nonzero = (E_k > 0)  & (bin_centers > 0)

    for kc, Ek in zip(k_centers[nonzero], E_k[nonzero]):
        energy_records.append({
            'run_idx': run_idx,
            'k_center': kc,
            'E_k': Ek
        })

    seps, flats = compute_flatness(u, max_sep, L)
    for sep, flat in zip(seps, flats):
        flat_records.append({
            'run_idx': run_idx,
            'separation': sep,
            'flatness': flat
        })

# %% Save results
df_flatness = pd.DataFrame(flat_records)
df_energy = pd.DataFrame(energy_records)

df_flatness.to_csv('./data/flatness_stats10_0.csv', index=False)
df_energy.to_csv('./data/energy_spectra10_0.csv', index=False)