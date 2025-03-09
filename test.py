import sys
import numpy as np
from numba import njit
import pygorpho as pg


in_name = sys.argv[1]
nar = np.load(in_name)
    

# @njit(parallel=True)
def main(nar, in_name):
    og_nar =        np.copy(nar)
    og_void_nar =   np.copy(nar)
    out_name_wo_Voids =      in_name.replace('.npy', '_wo_Voids.npy')
    out_name_just_dilated_voids =    in_name.replace('.npy', '_just_Dilated_Voids.npy')
    out_name_just_og_voids =    in_name.replace('.npy', '_just_OG_Voids.npy')

    # 1) Make nparray binary and send through Erosion and Dilation
    # This will become the Dilated Void layer
    print(np.unique(nar))
    nar[nar == 1] = 0 # No Core
    # nar[nar == 3] = 1 # Replace Void w/ 1 # nvm, as long as there are only 2 values the next step is fine

    strel_size = 3
    strel = np.ones((strel_size, strel_size, strel_size))
    nar = pg.gen.dilate(nar, strel)
    strel_size = 6
    strel = np.ones((strel_size, strel_size, strel_size))
    nar = pg.gen.erode(nar, strel)
    strel_size = 3
    strel = np.ones((strel_size, strel_size, strel_size))
    nar = pg.gen.dilate(nar, strel)

    nar -= 1 # Erosion and Dilation increment the values for some dumn reason
    print(np.unique(nar))



    # 2) Remove all Void data from original Core layer
    og_void_nar[og_void_nar == 1] = 0


    # 3) Create layer of original Void
    og_nar[og_nar == 3] = 1 # replace OG Void with Core

    
    with open(out_name_just_dilated_voids, 'wb+') as out_f:
        np.save(out_name_just_dilated_voids, nar)

    with open(out_name_wo_Voids, 'wb+') as out_f:
        np.save(out_name_wo_Voids, og_nar)
    
    with open(out_name_just_og_voids, 'wb+') as out_f:
        np.save(out_name_just_og_voids, og_void_nar)
        
main(nar, in_name)