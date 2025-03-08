import napari
from qtpy.QtWidgets import QComboBox

Boxname = 'ImageIDBox'
class demo(object):
    def __init__(self):
        self.viewer = napari.Viewer()
        imageidbox = QComboBox()   
        imageidbox.addItem(Boxname)
        self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left') 
        napari.run()
if __name__ == "__main__":
     demo()
