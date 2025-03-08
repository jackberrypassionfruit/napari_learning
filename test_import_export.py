from pathlib import Path
import numpy as np
from magicgui import magicgui
from skimage.io import imsave
import napari

import os

# from data_management_tools.database_connection import DBHelper
# from etl import extract, transform, load

import sys

data_path = Path(sys.argv[1])
annoted_path = data_path / 'annotated'
os.makedirs(annoted_path, exist_ok=True)

# with DBHelper('DCAP') as db:
#     cores_in_wip_patch = extract.get_cores_in_wip_patch(db)

def downsample_npy(nar_unreduced: np.ndarray, downsample_scale: int) -> np.ndarray:
    nar = nar_unreduced[
        ::downsample_scale, 
        ::downsample_scale, 
        ::downsample_scale
    ]
    return nar

def mark_layer_as_editted(event):
    # print(f"{event.source.name} changed its data!")
    
    active_layer = viewer.layers.selection.active
    active_layer.name = active_layer.name.replace('New', 'Editting').replace('Exported', 'Editting')
            
@magicgui(
    call_button =   "Add Core Image",
    part_id =       { "widget_type": "LineEdit" },
    layout =        "vertical"
)
def add_core_image(
    Instructions:   str = 'Scan barcode',
    part_id:        str = 'SR-24417_PN-9341028G_SN-25000-008A_INFERENCE_No_Air_just_Dilated_Voids' # TESTING ''
):
    first_nar_path = list(data_path.glob(f"*{part_id}*.npy"))[0]
    nar_unreduced = np.load(first_nar_path).astype(np.uint8)
    nar = nar_unreduced # if no downsampling needed, because Napari knows how to handle its memery
    # nar = downsample_npy(nar_unreduced, 2)
    viewer.add_labels(nar, name=f"(New) Core_{part_id}")
    
    active_layer = viewer.layers.selection.active
    active_layer.events.paint.connect(mark_layer_as_editted)
    
    processed_nar_path = first_nar_path.parent / 'annotated' / (first_nar_path.stem + "_Annotated" + first_nar_path.suffix)
    setattr(viewer.layers.selection.active, 'processed_path', processed_nar_path)


@magicgui(
    call_button =   "Export Core Image"
)
def export_core_image() -> None:
    active_layer = viewer.layers.selection.active
    active_layer.name = active_layer.name.replace('New', 'Exported').replace('Editting', 'Exported')
    
    active_layer_image = active_layer.data
    active_layer_processed_path = active_layer.processed_path
    with open(active_layer_processed_path, 'wb') as f:
        np.save(f, active_layer_image)
        
   
@magicgui(
    auto_call =  True,
    zoom_level = {"widget_type": "FloatSlider",  'max': 10},
    layout =     "vertical"
)
def set_zoom_level(zoom_level: "float"):
    viewer.camera.zoom = zoom_level
        
        
# Create the viewer (add an image later)
viewer = napari.Viewer()
# viewer, image_layer = napari.imshow(blobs.astype(float), name='blobs')

# Add widgets to viewer
viewer.window.add_dock_widget(add_core_image)
viewer.window.add_dock_widget(export_core_image)
viewer.window.add_dock_widget(set_zoom_level)

# Set the viewer to 3D
viewer.dims.ndisplay = 3

# Start napari Event Loop
napari.run()