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
max_sep  = 128


# %% Main loop
flat_records   = []

for j in range(num_triplets):
    print(f"Loading run {j}...", flush=True)
    Ax = tntt.load(os.path.join(folder, f"Ax_run{j}.TT"))
    Ay = tntt.load(os.path.join(folder, f"Ay_run{j}.TT"))
    Az = tntt.load(os.path.join(folder, f"Az_run{j}.TT"))

    A_x = z_order_to_normal_nopara(Ax.full().cpu().numpy().ravel(), np.zeros((N, N, N)), N, N, N)
    A_y = z_order_to_normal_nopara(Ay.full().cpu().numpy().ravel(), np.zeros((N, N, N)), N, N, N)
    A_z = z_order_to_normal_nopara(Az.full().cpu().numpy().ravel(), np.zeros((N, N, N)), N, N, N)

    u, v, w = compute_velocity_from_vector_potential(A_x, A_y, A_z, L)

    seps, flats = compute_flatness(u, max_sep, L)

    for sep, flat in zip(seps, flats):
        flat_records.append({
            'run_idx': j,
            'separation': sep,
            'flatness': flat
        })
# %% Save results
df_flat = pd.DataFrame(flat_records)

df_flat.to_csv('./data/flatnessR0.csv', index=False)
print("done", flush=True)