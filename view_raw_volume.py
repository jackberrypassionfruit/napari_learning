import image_operations as io
import napari
import sys

in_f = sys.argv[1]

with io.ImageOperations(in_f) as pp:
	volume = pp.array_to_volume(pp.array_1D)

viewer = napari.view_image(volume)
napari.run()
