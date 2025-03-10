
import sys
import numpy as np
import pygorpho as pg
import pandas as pd
from skimage.measure import label, regionprops_table
import napari

def strel_tuple(strel_size):
    return np.ones((strel_size, strel_size, strel_size))

# 0) initialize each ndarray in memory
in_name = sys.argv[1]
nar = np.load(in_name).astype(np.uint8)

nar[nar == 2] = 0 # No Air
og_nar =        np.copy(nar)
og_void_nar =   np.copy(nar)

# 1) Make nar binary and send through Erosion and Dilation
# This will become the Dilated Void layer
nar[nar == 1] = 0 # No Core

nar = pg.gen.dilate(nar, strel_tuple(3))
nar = pg.gen.erode(nar,  strel_tuple(6))
nar = pg.gen.dilate(nar, strel_tuple(3))

nar -= 1 # Erosion and Dilation increment all values for some dumb reason


# 2) Remove all Core data from original Core layer
og_void_nar[og_void_nar == 1] = 0


# 3) Create layer of only original Void
og_nar[og_nar == 3] = 1 # replace OG Void with Core


# 4) Get info on the dilated void layer
label_img = label(nar)
props_dict = regionprops_table(
    label_img,
    properties=('centroid', 'area'),
)
print(pd.DataFrame(props_dict))


# 5) Create the Napari viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3

viewer.add_labels(nar,          name='Only_Dilated_Voids')
viewer.add_labels(og_void_nar,  name='Only_OG_Voids')
viewer.add_labels(og_nar,       name='No_Voids')


# Start napari Event Loop
napari.run()
