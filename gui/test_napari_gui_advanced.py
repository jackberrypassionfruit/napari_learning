import napari
import skimage.data
import skimage.filters
from napari.types import ImageData

from magicgui import magicgui


# turn the gaussian blur function into a magicgui
# - 'auto_call' tells magicgui to call the function when a parameter changes
# - we use 'widget_type' to override the default "float" widget on sigma,
#   and provide a maximum valid value.
# - we contstrain the possible choices for 'mode'
@magicgui(
    auto_call=True,
    sigma={"widget_type": "FloatSlider", "max": 6},
    mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
    layout="veritical",
)
def gaussian_blur(layer: ImageData, sigma: float = 1.0, mode="nearest") -> ImageData:
    # Apply a gaussian blur to 'layer'.
    if layer is not None:
        return skimage.filters.gaussian(layer, sigma=sigma, mode=mode)

# create a viewer and add some images
viewer = napari.Viewer()
viewer.add_image(skimage.data.astronaut().mean(-1), name="astronaut")
viewer.add_image(skimage.data.grass().astype("float"), name="grass")

# Add it to the napari viewer
viewer.window.add_dock_widget(gaussian_blur)
# update the layer dropdown menu when the layer list changes
viewer.layers.events.changed.connect(gaussian_blur.reset_choices)

napari.run()
