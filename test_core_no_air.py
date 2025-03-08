import numpy as np
from scipy import ndimage as ndi

import napari
import sys

in_name = sys.argv[1]
nar = np.load(in_name)

viewer, image_layer = napari.imshow(nar, name='my_core')
labeled = ndi.label(nar)[0]
viewer.add_labels(labeled, name='my_core')

# Set the viewer to 3D
viewer.dims.ndisplay = 3

# start the event loop and show the viewer
napari.run()