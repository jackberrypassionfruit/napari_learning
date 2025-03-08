from magicgui import magicgui
import datetime
import pathlib

@magicgui(
    call_button=    "Calculate",
    slider_float=   {"widget_type": "FloatSlider",  'max': 10},
    dropdown=       {"choices":     ['first',   'second',   'third']},
)
def widget_demo(
    maybe:          bool,
    some_int:       int,
    spin_float=     3.14159,
    slider_float=   4.5,
    string=         "Text goes here",
    dropdown=       'first',
    date=           datetime.datetime.now(),
    filename=       pathlib.Path('/some/path.ext')
):
    ...

widget_demo.show()
input('Click ENTER to exit: ')