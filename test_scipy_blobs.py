from skimage import data
from scipy import ndimage as ndi

import napari


blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
viewer, image_layer = napari.imshow(blobs.astype(float), name='blobs')
labeled = ndi.label(blobs)[0]
viewer.add_labels(labeled, name='blob ID')

# Set the viewer to 3D
viewer.dims.ndisplay = 3

# start the event loop and show the viewer
napari.run()