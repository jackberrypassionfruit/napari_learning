from skimage import data
from skimage.util import img_as_float
from magicgui import magicgui
import napari

@magicgui(
    threshold = {
        "widget_type": "FloatSlider", 
        "max": 1
    }, 
    auto_call = True
)
def threshold_magic_widget(
    img_layer: "napari.layers.Image", 
    threshold: "float"
) -> "napari.types.LabelsData":
    return img_as_float(img_layer.data) > threshold

@magicgui( call_button="Import Astronaut" )
def add_astronaut_widget() -> None:
    viewer.add_image(data.astronaut(), name="astronaut")

# Create the viewer and add an image
viewer = napari.view_image(data.camera())
# Add widget to viewer
viewer.window.add_dock_widget(threshold_magic_widget)
viewer.window.add_dock_widget(add_astronaut_widget)
napari.run()