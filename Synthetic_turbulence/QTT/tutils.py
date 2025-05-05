import torch as tn
import torchtt as tntt
from scipy.special import comb
from scipy.optimize import fsolve
import numpy as np
import numba as nb


#build identity MPO
def I_qtt(d,dtype=tn.float64):

    return tntt.TT([tn.eye(2,dtype=dtype).reshape(1,2,2,1)]*d)

# build delta d_j(i) =0 if i!=j 1 otherwise

def dmps(i,n,dtype=tn.float64):
    bs = f"{i:0{n}b}"

    cores = []

    for j in range(n):
        cores.append( tn.tensor([[1,0] if bs[j] == '0' else [0,1]],dtype=dtype).reshape(1,2,1) )

    return tntt.TT(cores)

# build delta for mpo

def dmpo(i,j,n,dtype=tn.float64):
    bsm = f"{i:0{n}b}"
    bsn = f"{j:0{n}b}"

    cores = []

    for a in range(n):
        if (bsm[a] == '0' and bsn[a]=='0'):
            cores.append(tn.tensor( [[1,0],[0,0]] ,dtype=dtype).reshape(1,2,2,1) )
        
        elif (bsm[a] == '0' and bsn[a]=='1'):
            cores.append(tn.tensor( [[0,1],[0,0]] ,dtype=dtype).reshape(1,2,2,1) )
        
        elif (bsm[a] == '1' and bsn[a]=='0'):
            cores.append(tn.tensor( [[0,0],[1,0]] ,dtype=dtype).reshape(1,2,2,1) )

        else:
            cores.append(tn.tensor( [[0,0],[0,1]] ,dtype=dtype).reshape(1,2,2,1) )


    return tntt.TT(cores)


# build linear monomial
def X_qtt(d, dtype = tn.float64):

    # create first core

    c0 = tn.zeros([1,2,2], dtype=dtype)
    c0[:,0,:] = tn.tensor([1,0])
    c0[:,1,:] = tn.tensor([1,1/2])

    # create intermediate cores
    icores = []

    for i in range(1,d-1):
        ci = cl = tn.zeros([2,2,2], dtype=dtype)
        ci[:,0,:] = tn.eye(2)
        ci[:,1,:] = tn.tensor([[1,1/2**(i+1)],[0,1]])    

        icores.append(ci)
    #create last core

    cl = tn.zeros([2,2,1], dtype=dtype)
    cl[:,0,:] = tn.tensor([0,1]).reshape(2,1)
    cl[:,1,:] = tn.tensor([1/2**(d),1]).reshape(2,1)


    return tntt.TT([c0] + icores + [cl])

def qtt_polynomial_cores(a, d,dir='f',basis = 'm',dtype = tn.float64):
    """
    Constructs QTT cores for a polynomial M(x) = sum(a_k * x^k for k=0..p).
    
    Args:
        a (list): Polynomial coefficients [a_0, a_1, ..., a_p].
        d (int): Number of QTT dimensions (log2(grid size)).
    
    Returns:
        list: QTT cores as PyTorch tensors.
    """
    # Initialize QTT cores list
    cores = []
    p = len(a)-1

    #define the polynomial basis
    if basis == 'm':
        def phi(x,k):
            return pow(x,k)
    elif basis == 'a':
        a = tn.tensor(a,dtype=dtype)
        alpha = (pow(2,d)-1)/pow(2,d)/(p)
        def phi(x,k):
            return x*pow(x+alpha*k,k-1)

        m_list = []
        for k in range(1, p + 2):
            row = [0] * (k - 1)
            for i in range(0, p + 2 - k):
                b = comb(i + k - 1, i)
                term = b * pow(-(k - 1),i)*pow(alpha,i)
                row.append(term)

            m_list.append(row)

        m_list[0][0] = 1
        M = tn.tensor(m_list, dtype=dtype)
        #print(M)
        a = M @ a


    if dir == 'f':

        if d == 1:
            G1 = tn.zeros((2),dtype=dtype)
            G1[0] = a[0]
            G1[1] = sum(a[k] * phi(1/2,k) for k in range(0, p + 1))
            return G1

        # First core G1
        G1 = tn.zeros((1, 2, p + 1),dtype=dtype)  # Shape (1, 2, n+1)
        for s in range(p + 1):
            G1[0, 0, s] = a[s]
            G1[0, 1, s] = sum(a[k] * comb(k, s) * (phi(1/2,k-s)) for k in range(s, p + 1))
        cores.append(G1)

        # Intermediate cores G(x)
        for l in range(1, d - 1):
            G = tn.zeros((p + 1, 2, p + 1),dtype=dtype)  # Shape (n+1, 2, n+1)
            G[:, 0, :] = tn.eye(p+1,p+1) 
            for i in range(p + 1):
                for j in range(p + 1):
                    G[i, 1, j] = comb(i, i-j) * (phi(2**(-l-1),i-j)) if i >= j else 0
            cores.append(G)

        # Last core Gd
        Gd = tn.zeros((p + 1, 2, 1),dtype=dtype)  # Shape (n+1, 2, 1)
        v1 = tn.zeros(p+1)
        v1[0] = 1
        v2 = tn.tensor([1]+[phi(2**(-d),i) for i in range(1,p+1)])
        Gd[:, 0, 0] = v1
        Gd[:, 1, 0] = v2
        cores.append(Gd)

        return cores
    
    elif dir == 'b':
        # First core G1
        G1 = tn.zeros((1, 2, p + 1),dtype=dtype)  # Shape (1, 2, n+1)
        for s in range(p + 1):
            G1[0, 0, s] = a[s]
            G1[0, 1, s] = sum(a[k] * comb(k, s) * ( phi(2**(-d),k-s)  ) for k in range(s, p + 1))
        cores.append(G1)

        # Intermediate cores G(x)
        for l in range(1, d - 1):
            G = tn.zeros((p + 1, 2, p + 1),dtype=dtype)  # Shape (n+1, 2, n+1)
            G[:, 0, :] = tn.eye(p+1,p+1) 
            for i in range(p + 1):
                for j in range(p + 1):
                    G[i, 1, j] = comb(i, i-j) * ( phi(2**(-d+l),i-j) ) if i >= j else 0
            cores.append(G)

        # Last core Gd
        Gd = tn.zeros((p + 1, 2, 1),dtype=dtype)  # Shape (n+1, 2, 1)
        v1 = tn.zeros(p+1)
        v1[0] = 1
        v2 = tn.tensor([1]+[ phi(2**(-1),i)  for i in range(1,p+1)])
        Gd[:, 0, 0] = v1
        Gd[:, 1, 0] = v2
        cores.append(Gd)

        return [c.permute(2,1,0) for c in cores[::-1]]
    


    # Right shift matrix
def R_qtt( indx, d,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    #P = tn.tensor([[0,1],[1,0]],dtype=dtype)

    R = [ tn.stack([z, J],dim=1).reshape(1,2,2,2), tn.stack([J, I ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    bs = f"{indx:0{d}b}"

    cores = [R[0] if bs[0]=='0' else R[1]]

    for b in bs[1:-1]:

        cores.append(W[0] if b == '0' else W[1])

    
    cores.append(V[0] if bs[-1] == '0' else V[1])

    return tntt.TT(cores)

#left shift matrix
def L_qtt( indx, d,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    #P = tn.tensor([[0,1],[1,0]],dtype=dtype)

    Q = [ tn.stack([I, Jp],dim=1).reshape(1,2,2,2), tn.stack([Jp, z ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    bs = f"{indx:0{d}b}"

    cores = [Q[0] if bs[0]=='0' else Q[1]]

    for b in bs[1:-1]:

        cores.append(W[0] if b == '0' else W[1])

    
    cores.append(V[0] if bs[-1] == '0' else V[1])

    return tntt.TT(cores)

#Periodic shift matrix
def P_qtt( indx, d,dtype = tn.float64):

    z = tn.zeros((2,2),dtype=dtype)
    I = tn.eye(2,dtype=dtype)
    J = tn.tensor([[0,1],[0,0]],dtype=dtype)
    Jp = tn.tensor([[0,0],[1,0]],dtype=dtype)
    H = tn.tensor([[0,1],[1,0]],dtype=dtype)

    P = [ tn.stack([I, H],dim=1).reshape(1,2,2,2), tn.stack([H, I ],dim=1).reshape(1,2,2,2) ] 
    W = [ tn.stack([ tn.stack([I, Jp],dim=1), tn.stack([z,J],dim=1)]), tn.stack([ tn.stack([Jp, z],dim=1), tn.stack([J,I],dim=1) ]) ]
    V = [ tn.stack([I, z],dim=0).reshape(2,2,2,1), tn.stack([Jp, J ],dim=0).reshape(2,2,2,1) ]

    bs = f"{indx:0{d}b}"

    cores = [P[0] if bs[0]=='0' else P[1]]

    for b in bs[1:-1]:

        cores.append(W[0] if b == '0' else W[1])

    
    cores.append(V[0] if bs[-1] == '0' else V[1])

    return tntt.TT(cores)

def zkron(a,b):

    coresA = a.cores 
    coresB = b.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        try:
            zcores.append(tn.kron(coreA, coreB))
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e

    return tntt.TT(zcores)

def zkron3(a,b,c):

    coresA = a.cores 
    coresB = b.cores
    coresC = c.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        coreC = coresC[i].contiguous()
        try:
            zcores.append(tn.kron(tn.kron(coreA, coreB),coreC))
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e

    return tntt.TT(zcores)

def zukron(a,b):

    coresA = a.cores 
    coresB = b.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        try:
            m1 = tn.kron(coreA, tn.eye( int(coreB.shape[0]) , int(coreB.shape[0])).reshape(int(coreB.shape[0]) , 1,int(coreB.shape[0]) ) )
            m2 = tn.kron(tn.eye( int(coreA.shape[2]) , int(coreA.shape[2]) ).reshape(int(coreA.shape[2]) , 1, int(coreA.shape[2])),coreB )
            zcores.append(m1)
            zcores.append(m2)
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e
    return tntt.TT(zcores)

def zukron3(a,b,c):

    coresA = a.cores 
    coresB = b.cores
    coresC = c.cores

    l = len(coresA)
    zcores = []
    for i in range(len(coresA)):
        coreA = coresA[i].contiguous()  # Ensure contiguity
        coreB = coresB[i].contiguous()
        coreC = coresC[i].contiguous()
        try:
            rb = int(coreB.shape[0])
            rbr = int(coreB.shape[2])
            rc = int(coreC.shape[0])
            rar = int(coreA.shape[2]) 
            m1 = tn.kron( tn.kron(coreA, tn.eye(rb,rb).reshape(rb,1,rb) ) , tn.eye(rc, rc).reshape(rc,1,rc )  )
            m2 = tn.kron( tn.kron( tn.eye( rar, rar ).reshape(rar,1, rar),coreB ), tn.eye( rc,rc ).reshape(rc,1,rc) )
            m3 = tn.kron( tn.kron( tn.eye( rar, rar ).reshape(rar,1, rar),tn.eye( rbr,rbr ).reshape(rbr,1,rbr) ), coreC )
            zcores.append(m1)
            zcores.append(m2)
            zcores.append(m3)
        except RuntimeError as e:
            print(f"Error at index {i} with shapes {coreA.shape} and {coreB.shape}")
            raise e

    return tntt.TT(zcores)

def compute_morton_indices(n):
    """
    Compute a 2D tensor of Morton (Z-order) indices for a grid of size (2**n, 2**n).

    The Morton code for the coordinate (i, j) is given by interleaving the bits of i and j.
    
    Parameters:
        n (int): The number of bits, so that the grid size is (2**n, 2**n).
    
    Returns:
        tn.Tensor: A tensor of shape (2**n, 2**n) containing the Morton codes.
    """
    N = 2 ** n
    # Create a grid of row and column indices.
    # The 'indexing="ij"' keyword (available in recent PyTorch versions) ensures
    # that i corresponds to rows and j to columns.
    i, j = tn.meshgrid(tn.arange(N), tn.arange(N), indexing='ij')
    morton = tn.zeros((N, N), dtype=tn.int64)
    
    for bit in range(n):
        # Extract the bit at position 'bit' for i and j.
        # Then shift them to their appropriate locations in the interleaved number.
        morton |= (((i >> bit) & 1) << (2 * bit)) | (((j >> bit) & 1) << (2 * bit + 1))
    
    return morton

def izM(M_z, n):
    """
    Given a 2D tensor M_z of shape (2**n, 2**n) whose elements are arranged
    in Z‑order, return a new matrix with the original (row‐major) ordering.
    
    This is done by computing the inverse permutation of the Morton (Z‑order)
    mapping and applying it to the flattened data.
    
    Parameters:
        M_z (tn.Tensor): A 2D tensor of shape (2**n, 2**n) in Z‑order.
        n (int): Such that the matrix is of shape (2**n, 2**n).
    
    Returns:
        tn.Tensor: The matrix with the original ordering.
    """
    N = 2 ** n

    
    # Compute the Morton permutation.
    morton = compute_morton_indices(n)  # shape (N, N)
    perm = morton.flatten()  # perm is a tensor of length N*N.
    
    # Compute the inverse permutation.
    inv_perm = tn.empty_like(perm)
    inv_perm[perm] = tn.arange(perm.numel(), dtype=perm.dtype)
    
    M_z_flat = M_z
    # Apply the inverse permutation to recover the original flat ordering.
    M_orig_flat = M_z_flat[inv_perm]
    M_orig = M_orig_flat.reshape(N, N)
    return M_orig

def z_order_to_normal_torch(z_order_tensor, rows, cols):
    row_indices = tn.arange(rows, dtype=tn.int64)
    col_indices = tn.arange(cols, dtype=tn.int64)
    grid_rows, grid_cols = tn.meshgrid(row_indices, col_indices, indexing="ij")
    flat_rows = grid_rows.flatten()
    flat_cols = grid_cols.flatten()

    def part1by1(n):
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    morton = (part1by1(flat_rows) << 1) | part1by1(flat_cols)
    total = rows * cols
    normal_flat = tn.zeros(total, dtype=z_order_tensor.dtype, device=z_order_tensor.device)
    valid_mask = morton < z_order_tensor.numel()
    normal_flat[tn.nonzero(valid_mask).squeeze()] = z_order_tensor[morton[valid_mask]]
    return normal_flat.reshape(rows, cols)

def z_order_to_normal_3d(z_order_tensor, dim0, dim1, dim2):
    def part1by2(n):
        n &= 0x7FFF  # keep only 15 bits
        n = (n | (n << 32)) & 0x1F00000000FFFF
        n = (n | (n << 16)) & 0x1F0000FF0000FF
        n = (n | (n << 8))  & 0x100F00F00F00F00F
        n = (n | (n << 4))  & 0x10C30C30C30C30C3
        n = (n | (n << 2))  & 0x1249249249249249
        return n

    r = tn.arange(dim0, dtype=tn.int64, device=z_order_tensor.device)
    c = tn.arange(dim1, dtype=tn.int64, device=z_order_tensor.device)
    d = tn.arange(dim2, dtype=tn.int64, device=z_order_tensor.device)
    R, C, D = tn.meshgrid(r, c, d, indexing='ij')

    Rf = R.flatten()
    Cf = C.flatten()
    Df = D.flatten()

    morton = (part1by2(Rf) << 2) | (part1by2(Cf) << 1) | part1by2(Df)

    normal = tn.zeros((dim0, dim1, dim2), dtype=z_order_tensor.dtype, device=z_order_tensor.device)
    valid = morton < z_order_tensor.numel()
    normal.view(-1)[valid] = z_order_tensor[morton[valid]]

    return normal

@nb.njit      # no parallel=True
def z_order_to_normal_nopara(z_order, normal, dim0, dim1, dim2):
    n = z_order.size
    for i in range(n):
        code = i
        x = y = z = bit = 0
        while code:
            z |= (code & 1) << bit;  code >>= 1
            y |= (code & 1) << bit;  code >>= 1
            x |= (code & 1) << bit;  code >>= 1
            bit += 1

        if x < dim0 and y < dim1 and z < dim2:
            normal[x, y, z] = z_order[i]
    return normal

def interleave_bits_3d(x, y, z):
        """
        Interleave the bits of x, y, and z to produce a Morton (Z-order) index.
        The convention here is:
          - The bit from x goes to bit position (3*i + 2)
          - The bit from y goes to bit position (3*i + 1)
          - The bit from z goes to bit position (3*i)
        for each bit position i.
        """
        res = 0
        m = max(x.bit_length(), y.bit_length(), z.bit_length())
        for i in range(m):
            res |= ((x >> i) & 1) << (3 * i + 2)
            res |= ((y >> i) & 1) << (3 * i + 1)
            res |= ((z >> i) & 1) << (3 * i)
        return res
def z_order_to_normal_torch_3d(z_order_tensor, dim0, dim1, dim2):
    """
    Map a 3D array from Z-order (Morton order) to canonical (normal row-major) order using Pytn.

    Parameters:
        z_order_tensor (tn.Tensor): 1D tensor containing the 3D volume in Morton order.
        dim0 (int): Size along the first dimension.
        dim1 (int): Size along the second dimension.
        dim2 (int): Size along the third dimension.

    Returns:
        tn.Tensor: A 3D tensor of shape (dim0, dim1, dim2) in canonical order.
    """
    

    # Generate all (r, c, d) indices for the canonical tensor.
    r_indices = tn.arange(dim0, dtype=tn.int64)
    c_indices = tn.arange(dim1, dtype=tn.int64)
    d_indices = tn.arange(dim2, dtype=tn.int64)

    # Create a grid of indices using 'ij' indexing.
    grid_r, grid_c, grid_d = tn.meshgrid(r_indices, c_indices, d_indices, indexing="ij")
    
    # Flatten the grid arrays.
    flat_r = grid_r.flatten()
    flat_c = grid_c.flatten()
    flat_d = grid_d.flatten()

    # Compute the Morton (Z-order) index for each (r, c, d) coordinate.
    z_indices = tn.tensor(
        [interleave_bits_3d(r.item(), c.item(), d.item()) 
         for r, c, d in zip(flat_r, flat_c, flat_d)],
        dtype=tn.int64
    )

    # Create an empty tensor for the canonical (normal) ordering.
    normal_tensor = tn.zeros((dim0, dim1, dim2), dtype=z_order_tensor.dtype)

    # Total number of elements in the volume.
    total_elements = dim0 * dim1 * dim2

    # For each canonical index (from 0 to total_elements-1),
    # determine its (r, c, d) coordinate in the canonical order.
    # Then use the corresponding Morton index (if in bounds) to get the value
    # from the z_order_tensor.
    for idx, z_idx in enumerate(z_indices):
        # Ensure the Morton index is within bounds.
        if z_idx < z_order_tensor.numel():
            # Compute 3D indices from the canonical flattened index.
            r = idx // (dim1 * dim2)
            rem = idx % (dim1 * dim2)
            c = rem // dim2
            d = rem % dim2
            normal_tensor[r, c, d] = z_order_tensor[z_idx]

    return normal_tensor


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



def erank(ranks, dimensions):
    """
    Compute the effective rank (erank) of a TT representation.

    Parameters:
    - ranks (list): List of TT ranks, [r_0, r_1, ..., r_d].
    - dimensions (list): List of dimensions of the tensor, [n_1, n_2, ..., n_d].

    Returns:
    - float: The effective rank r_e.
    """
    d = len(dimensions)  # Number of dimensions

    # Compute the total number of parameters (S)
    S = sum(ranks[i] * dimensions[i] * ranks[i+1] for i in range(0, d - 1))

    # Define the function for fsolve
    def equation(re):
        return (
            re * dimensions[0]
            + sum(re**2 * dimensions[i] for i in range(1, d - 1))
            + re * dimensions[d - 1]
            - S
        )

    # Solve for r_e using fsolve
    r_e_initial_guess = np.mean(ranks)  # Initial guess for r_e
    r_e = fsolve(equation, r_e_initial_guess)[0]

    return r_e


def reduce(mps,i):
    cores = mps.cores
    new_cores = cores[:-2] + [tn.einsum('abc,cd -> abd',cores[-2],cores[-1][:,i,:])]

    return tntt.TT(new_cores)

def reduceg(mps,i,j):
    cores = mps.cores

    if j == -1:
        new_cores = cores[:j-1] + [tn.einsum('abc,cd -> abd',cores[j-1],cores[j][:,i,:])] 
    elif j != 0:
        new_cores = cores[:j-1] + [tn.einsum('abc,cd -> abd',cores[j-1],cores[j][:,i,:])] + cores[j+1:] 
    else:
        new_cores =  [tn.einsum('ac,cbd -> abd',cores[0][:,i,:], cores[1])] + cores[2:] 

    return tntt.TT(new_cores)

def connect(mps1,mps2,pd=3):

    if type(mps1) != list:
        cores1 = mps1.cores
    else:
        cores1 = mps1
        
    if type(mps2) != list:
        cores2 = mps2.cores
    else:
        cores2 = mps2

    cc = tn.einsum( 'ab,bc -> ac', cores1[-1].reshape(-1,pd), cores2[0].reshape(pd,-1))

    newcores = cores1[0:-1] + [ tn.einsum('ab, bcd -> acd',cc,cores2[1])] +  cores2[2:]

    return tntt.TT(newcores)


def hs(n_cores,t='1'):
    
    if t == '1':
        c1 = tntt.TT(tn.tensor([1,0],dtype=tn.float64))
        return tntt.kron(c1,tntt.ones([2]*(n_cores-1)))
    else:
        c1 = tntt.TT(tn.tensor([0,1],dtype=tn.float64))
        return tntt.kron(c1,tntt.ones([2]*(n_cores-1)))