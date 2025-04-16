# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

torch.set_default_dtype(torch.float64)


# %%
def stream_function_single_vortex(X, Y, Gamma=1.0, r0=0.1):
    """
    Generate a stream function for a single vortex on a 2D grid.
    Parameters:
      X, Y   : 2D arrays of coordinates.
      Gamma  : Circulation (strength) of the vortex.
      r0     : Core radius (smoothing parameter) to avoid singularity at r=0.
    Returns:
      psi    : 2D array of the stream function.
    """
    # Compute radial distance from the origin.
    r = np.sqrt(X**2 + Y**2)
    # Use a small epsilon to avoid log(0)
    epsilon = 1e-8
    psi = (Gamma / (2 * np.pi)) * np.log(1 + (r + epsilon) / r0)
    return psi


def generate_cascade_stream_function_overlapping(
    N=512, levels=None, seed=None, epsilon=0.1
):
    """
    Generate a 2D stream function field psi on an N x N grid using an overlapping cascade.

    Instead of updating disjoint blocks, at each cascade level a low-resolution random multiplier
    field is generated and then smoothly interpolated (using cubic interpolation) to the full grid.
    This ensures overlapping updates and smoother transitions.

    The multiplier field has a deterministic factor of 2^(-4/3) per level (to mimic psi increments
    ~ r^(4/3)) and random fluctuations controlled by epsilon.

    Parameters:
      N      : Grid size (must be a power of 2).
      levels : Number of cascade levels (default: log2(N)).
      seed   : Random seed for reproducibility.
      epsilon: Amplitude of random fluctuations (default 0.1).

    Returns:
      psi    : 2D array of the stream function.
    """
    if levels is None:
        levels = int(np.log2(N))
    if seed is not None:
        np.random.seed(seed)

    # Initialize the stream function uniformly.
    psi = np.ones((N, N), dtype=float)
    base_scaling = 2.0 ** (-4.0 / 3.0)

    # For each cascade level, generate a coarse multiplier field and interpolate it smoothly.
    for lvl in range(levels):
        # Create a coarse grid; resolution increases with cascade level.
        coarse_shape = (2**lvl, 2**lvl)
        # Random multipliers with mean 1.
        coarse_multiplier = 1 + (base_scaling**lvl) * epsilon * (
            np.random.rand(*coarse_shape) - 0.5
        )
        # Interpolate to full resolution using cubic interpolation.
        zoom_factor = (N / coarse_shape[0], N / coarse_shape[1])
        smooth_multiplier = zoom(
            coarse_multiplier,
            zoom_factor,
            order=2,
            # mode="nearest",
        )
        # Multiply the stream function by the smooth multiplier field.
        psi *= smooth_multiplier

    return psi


def generate_cascade_stream_function_vortices(
    N=512, levels=None, seed=None, epsilon=0.1
):
    """
    Generate a 2D stream function field psi on an N x N grid using an overlapping cascade.

    Instead of updating disjoint blocks, at each cascade level a low-resolution random multiplier
    field is generated and then smoothly interpolated (using cubic interpolation) to the full grid.
    This ensures overlapping updates and smoother transitions.

    The multiplier field has a deterministic factor of 2^(-4/3) per level (to mimic psi increments
    ~ r^(4/3)) and random fluctuations controlled by epsilon.

    Parameters:
      N      : Grid size (must be a power of 2).
      levels : Number of cascade levels (default: log2(N)).
      seed   : Random seed for reproducibility.
      epsilon: Amplitude of random fluctuations (default 0.1).

    Returns:
      psi    : 2D array of the stream function.
    """
    if levels is None:
        levels = int(np.log2(N))
    if seed is not None:
        np.random.seed(seed)

    # Initialize the stream function uniformly.
    # psi = np.ones((N, N), dtype=float)

    x = np.linspace(-np.pi, np.pi, N)
    X, Y = np.meshgrid(x, x)
    psi = stream_function_single_vortex(X, Y)*1e-1

    base_scaling = 2.0 ** (-4.0 / 3.0)

    # For each cascade level, generate a coarse multiplier field and interpolate it smoothly.
    for lvl in range(levels):
        # Create a coarse grid; resolution increases with cascade level.
        coarse_shape = 2**lvl
        x = np.linspace(-np.pi, np.pi, coarse_shape)
        X, Y = np.meshgrid(x, x)
        # Generate a random vortex at the center of the grid.
        vortex_coarse = stream_function_single_vortex(X, Y)
        # Random multipliers with mean 1.
        coarse_multiplier = 1 + (base_scaling**lvl) * (
            vortex_coarse + epsilon * (np.random.rand(coarse_shape, coarse_shape) - 0.5)
        )
        # Interpolate to full resolution.
        zoom_factor = (N / coarse_shape, N / coarse_shape)
        smooth_multiplier = zoom(
            coarse_multiplier,
            zoom_factor,
            order=2,
            # mode="nearest",
        )
        # Multiply the stream function by the smooth multiplier field.
        psi *= smooth_multiplier

    return psi


def compute_velocity_from_stream_function(psi, L=2 * np.pi):
    """
    Compute the velocity field (u,v) from the stream function ψ using central finite differences.
    u = dψ/dy, v = -dψ/dx.

    Parameters:
      psi : 2D array of the stream function.
      L   : Physical domain size (assumed square).

    Returns:
      u, v : 2D arrays of velocity components.
    """
    N = psi.shape[0]
    dx = L / N
    dy = L / N
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)

    # Use central differences in the interior (ignore boundaries for simplicity)
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)
    return u, v


def generate_divergence_free_kolmogorov_2d(
    N=128, L=2 * np.pi, exponent=-5 / 3, seed=None
):
    """
    Generate a 2D synthetic velocity field that is:
      - Divergence-free (incompressible).
      - Exhibits a power-law energy spectrum E(k) ~ k^(exponent).
        For a Kolmogorov-like spectrum, exponent = -5/3.

    Parameters:
      N        : Number of grid points in each spatial direction.
      L        : Physical domain size (assumed square).
      exponent : The desired power-law exponent for the energy spectrum.
                 (Kolmogorov 3D theory suggests -5/3, but we apply it in 2D here.)
      seed     : Optional random seed for reproducibility.

    Returns:
      u, v     : 2D arrays (N x N) of the velocity components in real space.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Set up wave numbers in Fourier space
    kx = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(KX**2 + KY**2)

    # Avoid division by zero at the k=0 mode
    k[0, 0] = 1.0

    # 2) Compute the amplitude for each mode so that E(k) ~ k^(exponent).
    #    For 2D, E(k) ~ k * |u(k)|^2. If E(k) ~ k^(exponent), then
    #    |u(k)|^2 ~ k^(exponent - 1) => |u(k)| ~ k^((exponent - 1)/2).
    #    For exponent = -5/3, that gives |u(k)| ~ k^(-4/3).
    amp = k ** ((exponent - 1) / 2.0)
    amp[0, 0] = 0.0  # remove mean flow (k=0)

    # 3) Generate random phases for preliminary fields
    phase_x = np.exp(1j * 2.0 * np.pi * np.random.rand(N, N))
    phase_y = np.exp(1j * 2.0 * np.pi * np.random.rand(N, N))

    # Preliminary velocity in Fourier space (unprojected)
    w_x = amp * phase_x
    w_y = amp * phase_y

    # 4) Enforce incompressibility (divergence-free condition) via projection
    #    u_hat = w_x - (KX (KX*w_x + KY*w_y)) / k^2
    #    v_hat = w_y - (KY (KX*w_x + KY*w_y)) / k^2
    #    but be careful with k=0 if needed (we already set it to zero amplitude).
    dot = KX * w_x + KY * w_y
    u_hat = w_x - (KX * dot) / (k**2)
    v_hat = w_y - (KY * dot) / (k**2)

    # 5) Inverse FFT to get the velocity field in real space
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real

    return u, v


# %%
N = 1024
L = 2 * np.pi
seed = 42
# %% Generate the cascade-based stream function

psi = generate_cascade_stream_function_overlapping(N=N, seed=seed, epsilon=0.5)

# Plot stream function
plt.figure(figsize=(8, 7))
plt.imshow(
    psi,
    origin="lower",
    extent=[0, 2 * np.pi, 0, 2 * np.pi],
    cmap="viridis",
)
plt.colorbar(label="Stream Function")
plt.title(
    "Cascade-Generated 2D Turbulence Snapshot\n(Divergence-free via Stream Function)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# %% Compute velocity field from ψ
u, v = compute_velocity_from_stream_function(psi, L=2 * np.pi)

uk, vk = generate_divergence_free_kolmogorov_2d(
    N=N,
    L=2 * np.pi,
    exponent=-5 / 3,
    seed=seed,
)
uk = uk * 1e5
vk = vk * 1e5

# %%
# Compute velocity magnitude
speed = np.sqrt(u**2 + v**2)
speedk = np.sqrt(uk**2 + vk**2)

# Plot velocity magnitude
plt.figure(figsize=(8, 7))
plt.imshow(speed, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="viridis")
plt.colorbar(label="Velocity magnitude")
plt.title(
    "Cascade-Generated 2D Turbulence Snapshot\n(Divergence-free via Stream Function)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# Plot velocity magnitude
plt.figure(figsize=(8, 7))
plt.imshow(speedk, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="viridis")
plt.colorbar(label="Velocity magnitude")
plt.title(
    "Fourier-Generated 2D Turbulence Snapshot\n(Divergence-free via Kolmogorov spectrum)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% Plot velocity streamlines
# x = np.linspace(0, 2 * np.pi, N)
# X, Y = np.meshgrid(x, x)

# plt.figure(figsize=(8, 7))
# plt.streamplot(
#     X,
#     Y,
#     u.T,
#     v.T,
#     density=1,
#     color=speed.T,
#     cmap="viridis",
#     broken_streamlines=False,
# )
# plt.colorbar(label="Velocity magnitude")
# plt.title(
#     "Cascade-Generated 2D Turbulence Snapshot\n(Divergence-free via Stream Function)"
# )
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# %% Compute the energy spectrum
# 1. Compute FFT
u_hat = np.fft.fft2(u)
v_hat = np.fft.fft2(v)
# 1. Compute FFT
uk_hat = np.fft.fft2(uk)
vk_hat = np.fft.fft2(vk)

# 2. Compute 2D energy density (example normalization)
# Adjust normalization based on your needs/FFT definition
E_hat_normalized = (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (N**4)
Ek_hat_normalized = (np.abs(uk_hat) ** 2 + np.abs(vk_hat) ** 2) / (N**4)

# 3. Get wavenumbers (kx, ky)
# Using fftfreq helps handle periodicity and shifting correctly
# Need to scale by 2*pi/L to get physical wavenumbers
k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)  # d is sample spacing
kx_grid, ky_grid = np.meshgrid(k, k)

# 4. Calculate wavenumber magnitude (k)
# Apply fftshift *before* calculating magnitudes if you want k=0 at the center
kx_shifted = np.fft.fftshift(kx_grid)
ky_shifted = np.fft.fftshift(ky_grid)
k_magnitude = np.sqrt(kx_shifted**2 + ky_shifted**2)

# Also shift the energy density to match the shifted wavenumbers
E_hat_shifted = np.fft.fftshift(E_hat_normalized)
Ek_hat_shifted = np.fft.fftshift(Ek_hat_normalized)

# 5. Radial Averaging (Binning)
# Define k bins (e.g., linear or log spacing)
k_min = 1.0  # Or slightly larger to avoid zero frequency if needed
k_max = np.max(k_magnitude) // np.sqrt(2)
num_bins = N // 2  # Example: up to Nyquist frequency
k_bins = np.linspace(k_min, k_max, num_bins + 1)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

E_k = np.zeros_like(k_centers)
E_k2 = np.zeros_like(k_centers)
counts = np.zeros_like(k_centers, dtype=int)
counts2 = np.zeros_like(k_centers, dtype=int)

# Loop through grid points (or use histogram/binned_statistic)
for i in range(N):
    for j in range(N):
        k_val = k_magnitude[i, j]
        # Find which bin k_val belongs to
        bin_index = np.searchsorted(k_bins, k_val) - 1
        if 0 <= bin_index < num_bins:
            E_k[bin_index] += E_hat_shifted[i, j]
            E_k2[bin_index] += Ek_hat_shifted[i, j]
            counts[bin_index] += 1

# Avoid division by zero for empty bins
valid_bins = counts > 0
E_k[valid_bins] /= counts[valid_bins]
E_k2[valid_bins] /= counts[valid_bins]

# E_k now holds the 1D radially averaged energy spectrum
# k_centers holds the corresponding wavenumber bin centers

# %% 6. Plot (e.g., using matplotlib)
plt.plot(
    np.log(k_centers[valid_bins]), np.log(E_k[valid_bins]), label="Stream Function"
)
plt.plot(np.log(k_centers[valid_bins]), np.log(E_k2[valid_bins]), label="Kolmogorov")

plt.xlabel("Wavenumber k")
plt.ylabel("Energy Spectrum E(k)")
plt.legend()
plt.show()

# %%
scaling = np.log2(E_k2[valid_bins]) / np.log2(k_centers[valid_bins])
plt.plot(k_centers[valid_bins], scaling)
plt.xlabel("Wavenumber k")
plt.ylabel("Scaling exponent")
plt.title("Scaling exponent of the energy spectrum")
plt.show()

print("Median scaling exponent:", np.median(scaling))
# %%
#################################################
# Stefano's code to compute the energy spectrum #
#################################################
N = 9
dim = 2**N

u, v = generate_divergence_free_kolmogorov_2d(
    N=dim,
    L=2 * np.pi,
    exponent=-5 / 3,
    seed=42,
)
# %%
dim = N
print(dim)
L = 2 * np.pi

uu_fft = np.fft.fft2(uk)
vv_fft = np.fft.fft2(vk)
print("fft done.")

uu_fft = (np.abs(uu_fft) / dim**2) ** 2
vv_fft = (np.abs(vv_fft) / dim**2) ** 2
print("fft normalized.")

k_end = int(dim / 2)
rx = np.array(range(dim)) - dim / 2 + 1
rx = np.roll(rx, int(dim / 2) + 1)

r = np.zeros((rx.shape[0], rx.shape[0]))
for i in range(rx.shape[0]):
    for j in range(rx.shape[0]):
        r[i, j] = rx[i] ** 2 + rx[j] ** 2
r = np.sqrt(r)

dx = 2 * np.pi / L
k = (np.array(range(k_end)) + 1) * dx

print("Start binning.")
bins = np.zeros((k.shape[0] + 1))
for N in range(k_end):
    if N == 0:
        bins[N] = 0
    else:
        bins[N] = (k[N] + k[N - 1]) / 2
bins[-1] = k[-1]

inds = torch.bucketize(torch.tensor(r * dx), torch.tensor(bins)).numpy()
spectrum = np.zeros((k.shape[0]))
bin_counter = np.zeros((k.shape[0]))

for N in range(k_end):
    spectrum[N] = np.sum(uu_fft[inds == N + 1]) + np.sum(vv_fft[inds == N + 1])
    bin_counter[N] = np.count_nonzero(inds == N + 1)

spectrum = spectrum * 2 * np.pi * (k**2) / (bin_counter * dx**2)
print("Spectrum obtained.")
# %%
plt.plot(np.log(k), np.log(spectrum))
plt.plot(np.log(k), (-5 / 3) * np.log(k) - 5, label="k^(-5/3)", linestyle="--")
plt.xlabel("log(k)")
plt.ylabel("log(E(k))")
plt.title("Energy spectrum")
# %%
