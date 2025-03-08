from skimage.measure import label

import sys
import numpy as np
import cupy as cp
import pygorpho as pg


in_name = sys.argv[1]
nar = cp.load(in_name)

# print(cp.unique(nar))

# out_name = 'new_fil.npy'
# out_name = in_name.replace('.npy', '_No_Air.npy')
out_name = in_name.replace('.npy', '_No_Core.npy')

strel = cp.ones((11, 11, 11))

labeled = label(nar.get())

# nar[nar == 2] = 0 # No Air
nar[nar == 1] = 0 # No Core
nar[nar == 3] = 1 # Replace Void w/ 1

print(f'{out_name=}')
with open(out_name, 'wb+') as out_f:
    cp.save(out_name, nar)

# print(f'{nar.max()=}')
# print(f'{np.unique(nar, axis=2)=}')
# print(f'{np.vstack(tuple(set(map(tuple,nar))))=}')

# print(f'{len(nar)=}')

# print(f'{np.unique(labeled[200:])=}')

