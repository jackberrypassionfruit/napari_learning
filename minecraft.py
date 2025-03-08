import numpy as np
from scipy import ndimage as ndi
from skimage.data import binary_blobs

import napari

blobs3d = binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)

blobs3dt = np.stack([np.roll(blobs3d, 3 * i, axis=2) for i in range(10)])

labels = ndi.label(blobs3dt[5])[0]

# viewer = napari.Viewer(ndisplay=3)

# image_layer = viewer.add_image(blobs3dt)
# labels_layer = viewer.add_labels(labels)
# viewer.dims.current_step = (5, 0, 0, 0)

print(labels)

# if __name__ == '__main__':
#     napari.run()