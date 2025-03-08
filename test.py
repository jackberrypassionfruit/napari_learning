import sys
import numpy as np
from numba import njit
import pygorpho as pg


in_name = sys.argv[1]
nar = np.load(in_name)
    

# @njit(parallel=True)
def main(nar, in_name):
    og_nar = np.copy(nar)
    og_void_nar = np.copy(nar)
    out_name = in_name.replace('.npy', '_wo_Voids.npy')
    out_name_2 = in_name.replace('.npy', '_just_Dilated_Voids.npy')
    out_name_3 = in_name.replace('.npy', '_just_OG_Voids.npy')

    og_void_nar[og_void_nar == 1] = 0


    nar[nar == 1] = 0 # No Core
    nar[nar == 3] = 1 # Replace Void w/ 1

    # print(np.unique(nar))
    strel_size = 11
    strel = np.ones((strel_size, strel_size, strel_size))
    nar = pg.gen.dilate(nar, strel)
    strel_size = 12
    strel = np.ones((strel_size, strel_size, strel_size))
    nar = pg.gen.erode(nar, strel)
    strel_size = 1
    strel = np.ones((strel_size, strel_size, strel_size))
    nar = pg.gen.dilate(nar, strel)

    # print(np.unique(nar))

    nar[nar == 1] = 0 # Null
    nar[nar == 2] = 3 # Void


    og_nar[og_nar == 3] = 1 # replace OG Void with Core
    
    # og_nar[nar == 3] = 3 # Void


    print(f'{out_name=}')
    with open(out_name, 'wb+') as out_f:
        np.save(out_name, og_nar)
    
    print(f'{out_name_2=}')
    with open(out_name_2, 'wb+') as out_f:
        np.save(out_name_2, nar)
    
    print(f'{out_name_3=}')
    with open(out_name_3, 'wb+') as out_f:
        np.save(out_name_3, og_void_nar)
        
main(nar, in_name)