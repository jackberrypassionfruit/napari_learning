from skimage import data
from scipy import ndimage as ndi
from magicgui import magicgui
import napari

from data_management_tools.database_connection import DBHelper
from etl import extract, transform, load

with DBHelper('DCAP') as db:
    cores_in_wip_patch = extract.get_cores_in_wip_patch(db)

@magicgui(
    call_button =   "Add Core Image",
    part_id =       { "widget_type": "LineEdit" },
    layout =        "vertical"
)
def add_core_image(
    Instructions:   str = 'Scan barcode',
    part_id:        str = ''
):
    if part_id in cores_in_wip_patch['part_id'].values:
        viewer.add_image(data.astronaut(), name=f"astronaut_{part_id}")
        ''' TODO
        make pull from real Core img file instead
        '''
    else:
        print(f'{part_id=} not in cores_in_wip_patch')

   
@magicgui(
    auto_call =  True,
    zoom_level = {"widget_type": "FloatSlider",  'max': 10},
    layout =     "vertical"
)
def set_zoom_level(zoom_level: "float"):
    viewer.camera.zoom = zoom_level
    
    
    
# Create the viewer and add a blobby image
blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
viewer = napari.view_image(blobs.astype(float), name='blobs')
# viewer, image_layer = napari.imshow(blobs.astype(float), name='blobs')

labeled = ndi.label(blobs)[0]
viewer.add_labels(labeled, name='blob ID')

# Add widgets to viewer
viewer.window.add_dock_widget(add_core_image)
viewer.window.add_dock_widget(set_zoom_level)

# Set the viewer to 3D
viewer.dims.ndisplay = 3

# Start napari Event Loop
napari.run()