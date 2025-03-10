
import sys
import gc
import numpy as np
import pygorpho as pg

import pandas as pd
from skimage.measure import label, regionprops_table

import napari

class DefectDefinition:
    def __init__(self, in_name):
        self.nar = np.load(in_name).astype(np.uint8)
        self.nar[self.nar == 2] = 0 # No Air
            
        self.og_nar =        np.copy(self.nar)
        self.og_void_nar =   np.copy(self.nar)
        
    
    def erode(self, strel_size):
        strel = np.ones((strel_size, strel_size, strel_size))
        self.nar = pg.gen.erode(self.nar, strel)
        
    def dilate(self, strel_size):
        strel = np.ones((strel_size, strel_size, strel_size))
        self.nar = pg.gen.dilate(self.nar, strel)

    def create_layers(self):
        

        self.nar[self.nar == 2] = 0 # No Air
            
        self.og_nar =        np.copy(self.nar)
        self.og_void_nar =   np.copy(self.nar)

        # 1) Make nparray binary and send through Erosion and Dilation
        # This will become the Dilated Void layer
        self.nar[self.nar == 1] = 0 # No Core
        # nar[nar == 3] = 1 # Replace Void w/ 1 # nvm, as long as there are only 2 values the next step is fine

        self.erode(3)
        self.dilate(3)
        self.erode(3)

        # strel_size = 3
        # strel = np.ones((strel_size, strel_size, strel_size))
        # nar = pg.gen.dilate(nar, strel)
        # strel_size = 6
        # strel = np.ones((strel_size, strel_size, strel_size))
        # nar = pg.gen.erode(nar, strel)
        # strel_size = 3
        # strel = np.ones((strel_size, strel_size, strel_size))
        # nar = pg.gen.dilate(nar, strel)

        self.nar -= 1 # Erosion and Dilation increment the values for some dumn reason


        # 2) Remove all Core data from original Core layer
        self.og_void_nar[self.og_void_nar == 1] = 0


        # 3) Create layer of only original Void
        self.og_nar[self.og_nar == 3] = 1 # replace OG Void with Core

    def print_dilated_layer_data(self):
        

        # Get info on the dilated void layer
        collected = gc.collect()
        print(collected)
        label_img = label(self.nar, connectivity=2)
        props_dict = regionprops_table(
            label_img,
            properties=('centroid', 'area'),
        )
        print(pd.DataFrame(props_dict))


    def run_napari_w_layers(self):
        # Create the Napari viewer
        viewer = napari.Viewer()

        # Set the viewer to 3D
        viewer.dims.ndisplay = 3


        viewer.add_labels(self.nar,          name='SN-25000-008A Only_Dilated_Voids')
        viewer.add_labels(self.og_void_nar,  name='SN-25000-008A Only_OG_Voids')
        viewer.add_labels(self.og_nar,       name='SN-25000-008A No_Voids')


        # Start napari Event Loop
        napari.run()

if __name__ == '__main__':
    defect_definition = DefectDefinition(sys.argv[1])
    
    defect_definition.create_layers()
    defect_definition.print_dilated_layer_data()
    defect_definition.run_napari_w_layers()