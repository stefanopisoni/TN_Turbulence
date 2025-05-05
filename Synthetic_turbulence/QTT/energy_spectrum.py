# %%
import pandas as pd
import numpy as np
from tutils import *
from cascade import *
import torchtt as tntt
import os

#%% data
folder = "./data/mps/10_0"
files = sorted(os.listdir(folder))
num_triplets = len(files) // 3
# %% Configuration
nscales  = 10
N        = 2**nscales
L        = 2 * np.pi

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

for j in range(num_triplets):
    print(f"Loading run {j}...", flush=True)
    Ax = tntt.load(os.path.join(folder, f"Ax_run{j}.TT"))
    Ay = tntt.load(os.path.join(folder, f"Ay_run{j}.TT"))
    Az = tntt.load(os.path.join(folder, f"Az_run{j}.TT"))

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
            'run_idx': j,
            'k_center': kc,
            'E_k': Ek
        })

# %% Save results
df_energy = pd.DataFrame(energy_records)

df_energy.to_csv('./data/energy_spectraR0.csv', index=False)