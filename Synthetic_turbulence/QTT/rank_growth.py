# %%
import pandas as pd
from tutils import *
from cascade import *

# %% Parameter Setup
dims, eranks, max_rank = [], [], []
samples_per_dim = 30
eps = 1e-8
nrank = 5
print("sim began")
# %% Data Collection
for i in range(5, 13):
    print("dimension" + str(i),flush=True)
    for _ in range(samples_per_dim // 3):
        A = gen_TT_cascade_3d(
            Nscales=i, nrank=nrank, levels=None, seed=None,
            epsilon=1, method='skc',
            eps=eps, boundary='periodic', order=1
        )
        for j in range(3):
            dims.append(i)
            eranks.append(erank(A[j].R, A[j].N))
            max_rank.append(max(A[j].R))

# %% Create DataFrame
data = pd.DataFrame({
    'Dimension': dims,
    'ERank': eranks,
    'MaxRank': max_rank
})

# %% Save to CSV
data.to_csv("./data/ranks5c1_1em8_p.csv", index=False)

print("done")