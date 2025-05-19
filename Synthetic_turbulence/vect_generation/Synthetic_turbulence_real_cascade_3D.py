import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

torch.set_default_dtype(torch.float64)


# %%
def vector_potential_single_vortex(X, Y, Z, Gamma=1.0, r0=0.1):
    """
    Generate a scalar vortex potential for a single vortex in 3D.
    This function is used to seed the 3D vector potential.
    Parameters:
      X, Y, Z : 3D arrays of coordinates.
      Gamma   : Circulation (strength) of the vortex.
      r0      : Core radius (smoothing parameter) to avoid singularity at r=0.
    Returns:
      phi     : 3D array of the vortex potential.
    """
    # Compute radial distance from the origin.
    r = np.sqrt(X**2 + Y**2 + Z**2)
    epsilon = 1e-8  # small epsilon to avoid log(0)
    # Note: In 3D the prefactor is adjusted (using 4*pi instead of 2*pi).
    phi = (Gamma / (4 * np.pi)) * np.log(1 + (r + epsilon) / r0)
    return phi


def generate_cascade_vector_potential_overlapping(N=128, levels=None, seed=None, epsilon=0.1):
    """
    Generate a 3D vector potential field A = (A_x, A_y, A_z) on an N x N x N grid using an overlapping cascade.

    Instead of updating disjoint blocks, at each cascade level a low-resolution random multiplier
    field is generated and then smoothly interpolated (using cubic interpolation) to the full grid.
    This ensures overlapping updates and smoother transitions.

    The multiplier field has a deterministic factor of 2^(-4/3) per level (to mimic increments
    with a 4/3 scaling) and random fluctuations controlled by epsilon.

    Parameters:
      N      : Grid size in each dimension (must be a power of 2).
      levels : Number of cascade levels (default: log2(N)).
      seed   : Random seed for reproducibility.
      epsilon: Amplitude of random fluctuations.

    Returns:
      A_x, A_y, A_z : 3D arrays representing the components of the vector potential.
    """
    if levels is None:
        levels = int(np.log2(N))
    if seed is not None:
        np.random.seed(seed)

    # Initialize each component uniformly.
    A_x = np.ones((N, N, N), dtype=float)
    A_y = np.ones((N, N, N), dtype=float)
    A_z = np.ones((N, N, N), dtype=float)
    base_scaling = 2.0 ** (-4.0 / 3.0)

    # For each cascade level, generate a coarse multiplier field and interpolate it smoothly.
    for lvl in range(levels):
        # Create a coarse grid; resolution increases with cascade level.
        coarse_shape = (2**lvl, 2**lvl, 2**lvl)
        # Random multipliers with mean 1.
        coarse_multiplier = 1 + (base_scaling**lvl) * epsilon * (np.random.rand(*coarse_shape) - 0.5)
        # Interpolate to full resolution using cubic (order=2) interpolation.
        zoom_factor = (N / coarse_shape[0], N / coarse_shape[1], N / coarse_shape[2])
        smooth_multiplier = zoom(coarse_multiplier, zoom_factor, order=2)
        # Multiply each component by the smooth multiplier field.
        A_x *= smooth_multiplier
        A_y *= smooth_multiplier
        A_z *= smooth_multiplier

    return A_x, A_y, A_z


def generate_cascade_vector_potential_vortices(N=128, levels=None, seed=None, epsilon=0.1):
    """
    Generate a 3D vector potential field A = (A_x, A_y, A_z) on an N x N x N grid using an overlapping cascade.
    In this version a coarse vortex potential is generated at each cascade level and used to modulate A.
    
    Parameters:
      N      : Grid size (must be a power of 2).
      levels : Number of cascade levels (default: log2(N)).
      seed   : Random seed for reproducibility.
      epsilon: Amplitude of random fluctuations.
      
    Returns:
      A_x, A_y, A_z : 3D arrays representing the components of the vector potential.
    """
    if levels is None:
        levels = int(np.log2(N))
    if seed is not None:
        np.random.seed(seed)

    # Define the full-resolution 3D grid.
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    z = np.linspace(-np.pi, np.pi, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # Initialize vector potential using the vortex potential.
    A_x = vector_potential_single_vortex(X, Y, Z) * 1e-1
    A_y = vector_potential_single_vortex(X, Y, Z) * 1e-1
    A_z = vector_potential_single_vortex(X, Y, Z) * 1e-1

    base_scaling = 2.0 ** (-4.0 / 3.0)

    # For each cascade level, generate a coarse vortex multiplier field and interpolate.
    for lvl in range(levels):
        # Create a coarse grid for the current cascade level.
        coarse_shape = 2**lvl  # scalar resolution in each dimension.
        x_coarse = np.linspace(-np.pi, np.pi, coarse_shape)
        y_coarse = np.linspace(-np.pi, np.pi, coarse_shape)
        z_coarse = np.linspace(-np.pi, np.pi, coarse_shape)
        Xc, Yc, Zc = np.meshgrid(x_coarse, y_coarse, z_coarse, indexing="ij")
        # Generate a coarse vortex potential.
        vortex_coarse = vector_potential_single_vortex(Xc, Yc, Zc)
        # Random multipliers with mean 1.
        coarse_multiplier = 1 + (base_scaling**lvl) * (vortex_coarse + epsilon * (np.random.rand(coarse_shape, coarse_shape, coarse_shape) - 0.5))
        # Interpolate to full resolution.
        zoom_factor = (N / coarse_shape, N / coarse_shape, N / coarse_shape)
        smooth_multiplier = zoom(coarse_multiplier, zoom_factor, order=2)
        # Multiply the vector potential components by the smooth multiplier field.
        A_x *= smooth_multiplier
        A_y *= smooth_multiplier
        A_z *= smooth_multiplier

    return A_x, A_y, A_z


def compute_velocity_from_vector_potential(A_x, A_y, A_z, L=2 * np.pi):
    """
    Compute the velocity field (u,v,w) from a vector potential A = (A_x, A_y, A_z)
    using central finite differences to approximate the curl.
    
    In 3D, the curl is given by:
      u = dA_z/dy - dA_y/dz,
      v = dA_x/dz - dA_z/dx,
      w = dA_y/dx - dA_x/dy.
    
    Parameters:
      A_x, A_y, A_z : 3D arrays for the vector potential components.
      L             : Physical domain size (assumed cubic: [0, L]^3).
      
    Returns:
      u, v, w       : 3D arrays of velocity components.
    """
    N = A_x.shape[0]
    dx = L / N  # assuming dx = dy = dz
    u = np.zeros_like(A_x)
    v = np.zeros_like(A_x)
    w = np.zeros_like(A_x)
    
    # Compute derivatives using central differences for interior points.
    # u = dA_z/dy - dA_y/dz
    u[1:-1, 1:-1, 1:-1] = ((A_z[1:-1, 2:, 1:-1] - A_z[1:-1, :-2, 1:-1]) -
                           (A_y[1:-1, 1:-1, 2:] - A_y[1:-1, 1:-1, :-2])) / (2 * dx)
    # v = dA_x/dz - dA_z/dx
    v[1:-1, 1:-1, 1:-1] = ((A_x[1:-1, 1:-1, 2:] - A_x[1:-1, 1:-1, :-2]) -
                           (A_z[2:, 1:-1, 1:-1] - A_z[:-2, 1:-1, 1:-1])) / (2 * dx)
    # w = dA_y/dx - dA_x/dy
    w[1:-1, 1:-1, 1:-1] = ((A_y[2:, 1:-1, 1:-1] - A_y[:-2, 1:-1, 1:-1]) -
                           (A_x[1:-1, 2:, 1:-1] - A_x[1:-1, :-2, 1:-1])) / (2 * dx)
    return u, v, w


def generate_divergence_free_kolmogorov_3d(N=128, L=2 * np.pi, exponent=-5/3, seed=None):
    """
    Generate a 3D synthetic velocity field that is:
      - Divergence-free (incompressible).
      - Exhibits a power-law energy spectrum E(k) ~ k^(exponent).
        For a Kolmogorov-like spectrum in 3D, noting that E(k) ~ k^2 |u(k)|^2,
        the Fourier amplitude scales as |u(k)| ~ k^((exponent-2)/2).
    
    Parameters:
      N        : Number of grid points in each spatial direction.
      L        : Physical domain size (assumed cubic).
      exponent : Desired power-law exponent for the energy spectrum.
      seed     : Optional random seed for reproducibility.
      
    Returns:
      u, v, w  : 3D arrays (N x N x N) of the velocity components in real space.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Set up wave numbers in Fourier space.
    kx = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    kz = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    k[0, 0, 0] = 1.0  # avoid division by zero

    # 2) Compute the amplitude for each Fourier mode.
    # For 3D, |u(k)| ~ k^((exponent-2)/2).
    amp = k ** ((exponent - 2) / 2.0)
    amp[0, 0, 0] = 0.0  # remove the k=0 (mean flow) mode

    # 3) Generate random phases for the three velocity components.
    phase_x = np.exp(1j * 2.0 * np.pi * np.random.rand(N, N, N))
    phase_y = np.exp(1j * 2.0 * np.pi * np.random.rand(N, N, N))
    phase_z = np.exp(1j * 2.0 * np.pi * np.random.rand(N, N, N))
    w_x = amp * phase_x
    w_y = amp * phase_y
    w_z = amp * phase_z

    # 4) Enforce the divergence-free condition via projection.
    dot = KX * w_x + KY * w_y + KZ * w_z
    u_hat = w_x - (KX * dot) / (k**2)
    v_hat = w_y - (KY * dot) / (k**2)
    w_hat = w_z - (KZ * dot) / (k**2)

    # 5) Inverse FFT to obtain the velocity field in real space.
    u = np.fft.ifftn(u_hat).real
    v = np.fft.ifftn(v_hat).real
    w = np.fft.ifftn(w_hat).real

    return u, v, w


# %%
# Main parameters
N = 256
L = 2 * np.pi
seed = 42

# %% Generate the cascade-based vector potential (analogous to the 2D stream function)
A_x, A_y, A_z = generate_cascade_vector_potential_overlapping(N=N, seed=seed, epsilon=0.5)

# Plot one mid-plane slice (here A_z) of the vector potential
plt.figure(figsize=(8, 7))
plt.imshow(
    A_z[:, :, N//2],
    origin="lower",
    extent=[0, 2 * np.pi, 0, 2 * np.pi],
    cmap="viridis",
)
plt.colorbar(label="Vector Potential A_z")
plt.title(
    "Cascade-Generated 3D Turbulence Snapshot\n(Vector Potential A_z, mid-plane)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% Compute velocity field from the vector potential (via its curl)
u, v, w = compute_velocity_from_vector_potential(A_x, A_y, A_z, L=2 * np.pi)

# Generate the divergence-free 3D turbulence field via Fourier projection
uk, vk, wk = generate_divergence_free_kolmogorov_3d(
    N=N,
    L=2 * np.pi,
    exponent=-5 / 3,
    seed=seed,
)
uk = uk * 1e5
vk = vk * 1e5
wk = wk * 1e5

# %%
# Compute velocity magnitudes
speed = np.sqrt(u**2 + v**2 + w**2)
speedk = np.sqrt(uk**2 + vk**2 + wk**2)

# Plot velocity magnitude for cascade-generated field (mid-plane slice)
plt.figure(figsize=(8, 7))
plt.imshow(speed[:, :, N//2], origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="viridis")
plt.colorbar(label="Velocity magnitude")
plt.title(
    "Cascade-Generated 3D Turbulence Snapshot\n(Velocity magnitude, mid-plane)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot velocity magnitude for Fourier-generated field (mid-plane slice)
plt.figure(figsize=(8, 7))
plt.imshow(speedk[:, :, N//2], origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="viridis")
plt.colorbar(label="Velocity magnitude")
plt.title(
    "Fourier-Generated 3D Turbulence Snapshot\n(Velocity magnitude, mid-plane)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% Compute the energy spectrum
# 1. Compute FFTs of the cascade-based and Fourier-generated velocity fields.
u_hat = np.fft.fftn(u)
v_hat = np.fft.fftn(v)
w_hat = np.fft.fftn(w)

uk_hat = np.fft.fftn(uk)
vk_hat = np.fft.fftn(vk)
wk_hat = np.fft.fftn(wk)

# 2. Compute 3D energy density (normalization adjusted for 3D; N^6 factor since volume ~ N^3 and energy ~ square)
E_hat_normalized = (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2) / (N**6)
Ek_hat_normalized = (np.abs(uk_hat)**2 + np.abs(vk_hat)**2 + np.abs(wk_hat)**2) / (N**6)

# 3. Get wavenumbers in 3D.
kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
kz = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
KX_shifted = np.fft.fftshift(KX)
KY_shifted = np.fft.fftshift(KY)
KZ_shifted = np.fft.fftshift(KZ)
k_magnitude = np.sqrt(KX_shifted**2 + KY_shifted**2 + KZ_shifted**2)

# Shift the energy densities accordingly.
E_hat_shifted = np.fft.fftshift(E_hat_normalized)
Ek_hat_shifted = np.fft.fftshift(Ek_hat_normalized)

# 4. Radial Averaging (Binning in spherical shells)
# Define k bins. Here we set a minimum k=1.0 and maximum k using the maximum k divided by sqrt(3) for 3D.
k_min = 1.0
k_max = np.max(k_magnitude) // np.sqrt(3)
num_bins = N // 2  # Example: up to (roughly) the Nyquist frequency in 3D.
k_bins = np.linspace(k_min, k_max, num_bins + 1)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

# Initialize accumulators for energy and counts.
E_k = np.zeros_like(k_centers)
E_k2 = np.zeros_like(k_centers)
counts = np.zeros_like(k_centers, dtype=int)
counts2 = np.zeros_like(k_centers, dtype=int)

# Loop over all grid points in the 3D FFT result.
# (For clarity, we flatten the arrays and loop over the flattened index.)
k_mag_flat = k_magnitude.flatten()
E_flat = E_hat_shifted.flatten()
Ek_flat = Ek_hat_shifted.flatten()
for idx in range(k_mag_flat.size):
    k_val = k_mag_flat[idx]
    bin_index = np.searchsorted(k_bins, k_val) - 1
    if 0 <= bin_index < num_bins:
        E_k[bin_index] += E_flat[idx]
        E_k2[bin_index] += Ek_flat[idx]
        counts[bin_index] += 1

# Avoid division by zero by averaging only in bins that received contributions.
valid_bins = counts > 0
E_k[valid_bins] /= counts[valid_bins]
E_k2[valid_bins] /= counts[valid_bins]

# 5. Plot the energy spectra (in log-log space).
plt.figure(figsize=(8, 6))
plt.plot(np.log(k_centers[valid_bins]), np.log(E_k[valid_bins]), label="Vector Potential")
plt.plot(np.log(k_centers[valid_bins]), np.log(E_k2[valid_bins]), label="Kolmogorov (Fourier)")
plt.xlabel("log(k)")
plt.ylabel("log(E(k))")
plt.legend()
plt.show()

# 6. Plot the scaling exponent.
scaling = np.log2(E_k2[valid_bins]) / np.log2(k_centers[valid_bins])
plt.figure(figsize=(8, 6))
plt.plot(k_centers[valid_bins], scaling)
plt.xlabel("Wavenumber k")
plt.ylabel("Scaling exponent")
plt.title("Scaling exponent of the energy spectrum (3D)")
plt.show()

print("Median scaling exponent:", np.median(scaling))



# %% Compute and plot the flatness for both approaches in log-log scale

def compute_flatness(u_field, max_sep, L):
    """
    Compute the flatness (kurtosis) of velocity increments in the x-direction for a 3D field.
    
    Parameters:
      u_field : 3D array (velocity component u).
      max_sep : Maximum separation (in grid points) to compute the increments.
      L       : Physical domain size.
    
    Returns:
      separations : 1D array of physical separation distances.
      flatness    : 1D array of flatness values at each separation.
    """
    N = u_field.shape[0]
    dx = L / N
    separations = np.arange(1, max_sep+1) * dx
    flatness = []
    for dr in range(1, max_sep+1):
        # Compute increments in the x-direction for all valid grid points.
        diff = u_field[dr:, :, :] - u_field[:-dr, :, :]
        # diff = u_field[dr:-1, :, :] - u_field[1:-dr, :, :]
        S2 = np.mean(diff**2)
        S4 = np.mean(diff**4)
        flatness.append(S4 / (S2**2))
    return separations, np.array(flatness)

# Set maximum separation (in grid points) for structure function analysis.
max_sep = 128

# Compute flatness for the cascade-based velocity field (u component)
sep_cascade, flatness_cascade = compute_flatness(u, max_sep, L)

# Compute flatness for the Fourier-based (Kolmogorov) velocity field (u component)
sep_fourier, flatness_fourier = compute_flatness(uk, max_sep, L)

# Plot the flatness versus separation for both fields in log-log scale.
plt.figure(figsize=(8, 6))
plt.loglog(sep_cascade, flatness_cascade, label="Cascade (Vector Potential)")
plt.loglog(sep_fourier, flatness_fourier, label="Fourier (Kolmogorov)")
plt.xlabel("Separation (physical units)")
plt.ylabel("Flatness (kurtosis)")
plt.title("Flatness of u–velocity increments vs Separation (log-log scale)")
plt.legend()
plt.show()
# %%
