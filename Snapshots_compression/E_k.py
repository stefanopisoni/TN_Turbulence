# %% [markdown]
# Load the data obtained from Madrid database and compute correlations.
import sys
print(sys.path)
# %%
import numpy as np
sys.path.append('../')
import torch
from tensnet import *
# torch.set_default_device('cuda')

torch.pi = torch.acos(torch.zeros(1)).item() * 2

# encoding = 'sta'
# BD_MAX = 10000

# component = 'u'
# ID = f'tur_{component}_mps_{encoding}_BD_{BD_MAX}_torch'
# u = torch.load(f'../Snapshots/{ID}.tens')
# print("Success imported and loaded:",f'{ID}')

# component = 'v'
# ID = f'tur_{component}_mps_{encoding}_BD_{BD_MAX}_torch'
# v = torch.load(f'../Snapshots/{ID}.tens')
# print("Success imported and loaded:",f'{ID}')

# component = 'w'
# ID = f'tur_{component}_mps_{encoding}_BD_{BD_MAX}_torch'
# w = torch.load(f'../Snapshots/{ID}.tens')
# print("Success imported and loaded:",f'{ID}')

u = torch.load(f'../Snapshots/tur_u.tens')
v = torch.load(f'../Snapshots/tur_v.tens')
w = torch.load(f'../Snapshots/tur_w.tens')


N = 10
skip=2
dim=int(round(2**N/skip))
print(dim)
L=2*torch.pi

uu_fft=torch.fft.fftn(u[::skip,::skip,::skip])
vv_fft=torch.fft.fftn(v[::skip,::skip,::skip])
ww_fft=torch.fft.fftn(w[::skip,::skip,::skip])
print('fft done.')

uu_fft=(torch.abs(uu_fft)/dim**3)**2
vv_fft=(torch.abs(vv_fft)/dim**3)**2
ww_fft=(torch.abs(ww_fft)/dim**3)**2
print('fft normalized.')

k_end=int(dim/2)
rx=torch.Tensor(range(dim))-dim/2+1
rx=torch.roll(rx,int(dim/2)+1)

r=torch.zeros((rx.shape[0],rx.shape[0],rx.shape[0]))
for i in range(rx.shape[0]):
    for j in range(rx.shape[0]):
            r[i,j,:]=rx[i]**2+rx[j]**2+rx[:]**2
r=torch.sqrt(r)

dx=2*torch.pi/L
k=(torch.tensor(range(k_end))+1)*dx

print('Start binning.')
bins=torch.zeros((k.shape[0]+1))
for N in range(k_end):
    if N==0:
        bins[N]=0
    else:
        bins[N]=(k[N]+k[N-1])/2    
bins[-1]=k[-1]

inds = torch.bucketize(r*dx, bins)
spectrum=torch.zeros((k.shape[0]))
bin_counter=torch.zeros((k.shape[0]))

for N in range(k_end):
    spectrum[N]=torch.sum(uu_fft[inds==N+1])+torch.sum(vv_fft[inds==N+1])+torch.sum(ww_fft[inds==N+1])
    bin_counter[N]=torch.count_nonzero(inds==N+1)

spectrum=spectrum*2*torch.pi*(k**2)/(bin_counter*dx**3)
print('Spectrum obtained.')

# torch.save(spectrum, f'E_k_{encoding}_{BD_MAX}.tens')
torch.save(spectrum, f'E_k_original.tens')