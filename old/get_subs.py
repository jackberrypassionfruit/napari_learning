import os

folder = r'C:\Users\JPASHAYAN\repos\napari_learning\napari-env-prod\Lib\site-packages'
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

for row in [row for row in subfolders if not row.endswith('dist-info') and not row.endswith('egg-info')]:
    print(row.split('\\')[-1])
    

('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\adodbapi', 'adodbapi'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\alabaster', 'alabaster'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\altgraph', 'altgraph'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\annotated_types', 'annotated_types'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\app_model', 'app_model'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\asciitree', 'asciitree'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\asttokens', 'asttokens'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\attr', 'attr'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\attrs', 'attrs'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\babel', 'babel'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\build', 'build'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\cachey', 'cachey'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\certifi', 'certifi'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\cffi', 'cffi'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\charset_normalizer', 'charset_normalizer'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\click', 'click'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\cloudpickle', 'cloudpickle'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\colorama', 'colorama'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\comm', 'comm'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\cytoolz', 'cytoolz'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\dask', 'dask'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\dateutil', 'dateutil'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\debugpy', 'debugpy'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\docstring_parser', 'docstring_parser'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\docutils', 'docutils'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\exceptiongroup', 'exceptiongroup'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\executing', 'executing'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\fasteners', 'fasteners'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\flexcache', 'flexcache'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\flexparser', 'flexparser'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\freetype', 'freetype'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\fsspec', 'fsspec'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\future', 'future'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\greenlet', 'greenlet'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\h2', 'h2'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\hpack', 'hpack'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\hyperframe', 'hyperframe'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\idna', 'idna'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\imagecodecs', 'imagecodecs'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\imageio', 'imageio'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\imagesize', 'imagesize'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\importlib_metadata', 'importlib_metadata'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\importlib_resources', 'importlib_resources'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\in_n_out', 'in_n_out'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\ipykernel', 'ipykernel'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\IPython', 'IPython'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\isapi', 'isapi'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\jedi', 'jedi'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\jinja2', 'jinja2'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\jsonschema', 'jsonschema'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\jsonschema_specifications', 'jsonschema_specifications'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\jupyter_client', 'jupyter_client'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\jupyter_core', 'jupyter_core'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\kiwisolver', 'kiwisolver'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\lazy_loader', 'lazy_loader'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\libfuturize', 'libfuturize'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\libpasteurize', 'libpasteurize'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\llvmlite', 'llvmlite'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\locket', 'locket'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\magicgui', 'magicgui'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\markdown_it', 'markdown_it'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\markupsafe', 'markupsafe'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\matplotlib_inline', 'matplotlib_inline'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\mdurl', 'mdurl'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\msgpack', 'msgpack'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\napari', 'napari'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\napari_builtins', 'napari_builtins'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\napari_console', 'napari_console'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\napari_plugin_engine', 'napari_plugin_engine'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\napari_plugin_manager', 'napari_plugin_manager'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\napari_svg', 'napari_svg'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\networkx', 'networkx'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\npe2', 'npe2'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\numba', 'numba'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\numcodecs', 'numcodecs'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\numpy', 'numpy'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\numpydoc', 'numpydoc'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\OpenGL', 'OpenGL'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\ordlookup', 'ordlookup'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\packaging', 'packaging'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pandas', 'pandas'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\parso', 'parso'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\partd', 'partd'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\past', 'past'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\PIL', 'PIL'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pint', 'pint'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pip', 'pip'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pkg_resources', 'pkg_resources'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\platformdirs', 'platformdirs'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\ply', 'ply'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pooch', 'pooch'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\prompt_toolkit', 'prompt_toolkit'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\psutil', 'psutil'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\psygnal', 'psygnal'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pure_eval', 'pure_eval'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pyconify', 'pyconify'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pycparser', 'pycparser'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pydantic', 'pydantic'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pydantic_compat', 'pydantic_compat'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pydantic_core', 'pydantic_core'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pygments', 'pygments'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\PyInstaller', 'PyInstaller'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pyproject_hooks', 'pyproject_hooks'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\PyQt5', 'PyQt5'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pythonwin', 'pythonwin'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pytz', 'pytz'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pywin32_system32', 'pywin32_system32'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\pywt', 'pywt'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\qtconsole', 'qtconsole'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\qtpy', 'qtpy'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\referencing', 'referencing'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\requests', 'requests'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\rich', 'rich'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\rpds', 'rpds'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\scipy', 'scipy'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\setuptools', 'setuptools'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\shellingham', 'shellingham'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\sipbuild', 'sipbuild'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\skimage', 'skimage'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\snowballstemmer', 'snowballstemmer'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\sphinx', 'sphinx'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\sphinxcontrib', 'sphinxcontrib'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\sqlalchemy', 'sqlalchemy'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\stack_data', 'stack_data'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\superqt', 'superqt'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tabulate', 'tabulate'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tifffile', 'tifffile'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tlz', 'tlz'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\toml', 'toml'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tomli', 'tomli'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tomli_w', 'tomli_w'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\toolz', 'toolz'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tornado', 'tornado'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tqdm', 'tqdm'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\traitlets', 'traitlets'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\typer', 'typer'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\tzdata', 'tzdata'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\urllib3', 'urllib3'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\vispy', 'vispy'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\wcwidth', 'wcwidth'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\wheel', 'wheel'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\win32', 'win32'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\win32com', 'win32com'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\win32comext', 'win32comext'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\win32ctypes', 'win32ctypes'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\wrapt', 'wrapt'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\yaml', 'yaml'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\zarr', 'zarr'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\zipp', 'zipp'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\zmq', 'zmq'),
('C:\\Users\\JPASHAYAN\\repos\\napari_learning\\napari-env-prod\\Lib\\site-packages\\zstandard', 'zstandard')











adodbapi
alabaster
altgraph
annotated_types
app_model
asciitree
asttokens
attr
attrs
babel
build
cachey
certifi
cffi
charset_normalizer
click
cloudpickle
colorama
comm
cytoolz
dask
dateutil
debugpy
docstring_parser
docutils
exceptiongroup
executing
fasteners
flexcache
flexparser
freetype
fsspec
future
greenlet
h2
hpack
hyperframe
idna
imagecodecs
imageio
imagesize
importlib_metadata
importlib_resources
in_n_out
ipykernel
IPython
isapi
jedi
jinja2
jsonschema
jsonschema_specifications
jupyter_client
jupyter_core
kiwisolver
lazy_loader
libfuturize
libpasteurize
llvmlite
locket
magicgui
markdown_it
markupsafe
matplotlib_inline
mdurl
msgpack
napari
napari_builtins
napari_console
napari_plugin_engine
napari_plugin_manager
napari_svg
networkx
npe2
numba
numcodecs
numpy
numpydoc
OpenGL
ordlookup
packaging
pandas
parso
partd
past
PIL
pint
pip
pkg_resources
platformdirs
ply
pooch
prompt_toolkit
psutil
psygnal
pure_eval
pyconify
pycparser
pydantic
pydantic_compat
pydantic_core
pygments
PyInstaller
pyproject_hooks
PyQt5
pythonwin
pytz
pywin32_system32
pywt
qtconsole
qtpy
referencing
requests
rich
rpds
scipy
setuptools
shellingham
sipbuild
skimage
snowballstemmer
sphinx
sphinxcontrib
sqlalchemy
stack_data
superqt
tabulate
tifffile
tlz
toml
tomli
tomli_w
toolz
tornado
tqdm
traitlets
typer
tzdata
urllib3
vispy
wcwidth
wheel
win32
win32com
win32comext
win32ctypes
wrapt
yaml
zarr
zipp
zmq
zstandard