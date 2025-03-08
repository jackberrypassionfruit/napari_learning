import numpy as np
# from scipy import ndimage as ndi
# from skimage.measure import label

import napari
import sys

DOWNSAMPLE_SCALE = 2

in_name = sys.argv[1]
nar_unreduced = np.load(in_name)

nar = nar_unreduced[
    ::DOWNSAMPLE_SCALE, 
    ::DOWNSAMPLE_SCALE, 
    ::DOWNSAMPLE_SCALE
]

# labeled = ndi.label(nar)[0].astype(np.uint8)
labeled = nar
# labeled = label(nar)
# viewer, image_layer = napari.imshow(nar, name='my_core_to_remove') # Invisible layer
# viewer = napari.view_image(nar, name='my_core_from_image')
viewer = napari.view_labels(labeled, name='my_core') # Create a napari viewer using just label

# viewer.layers['my_core_from_image'].visible = False
# viewer.add_labels(labeled, name='my_core') # Viewable layer
# del viewer.layers['my_core_from_image']


# Set the viewer to 3D
viewer.dims.ndisplay = 3


# Set the default brush size = 1
#   So that annotations appear on the surface, not sticking out
#   May need to do this every time a new part is imported
#   TODO Fixure out how to index the dict by name 
# viewer.window._qt_viewer.controls.widgets['my_core'].brush_size = 1
list(viewer.window._qt_viewer.controls.widgets)[-1].brush_size = 1
# print(vars(viewer.window._qt_viewer.controls.viewer).keys())

# start the event loop and show the viewer
napari.run()