
from typing import Union
import matplotlib as mpl
import matplotlib.font_manager as fm
import warnings
from datetime import datetime, timedelta
import os
import requests
from ..utils import console
from matplotlib import rcParams

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

name = "epione"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"


_has_printed_logo = False  # Flag to ensure logo prints only once


# emoji map for status reporting
EMOJI = {
    "start":        "🔬",  # experiment start
    "deps":         "🔗",  # dependency check
    "settings":     "⚙️",  # configure settings
    "warnings":     "🚫",  # suppress warnings
    "gpu":          "🧬",  # GPU check
    "logo":         "🌟",  # print logo
    "done":         "✅",  # done
}


def plot_set(verbosity: int = 3, dpi: int = 80, 
             facecolor: str = 'white', 
             font_path: str = None,
             ipython_format: str  = "retina",
             dpi_save: int = 300,
             transparent: bool = None,
             scanpy: bool = True,
             fontsize: int = 14,
             color_map: Union[str, None] = None,
             figsize: Union[int, None] = None,
             vector_friendly: bool = True,
             ):
    r"""Configure plotting settings for OmicVerse.
    
    Arguments:
        verbosity: Scanpy verbosity level. Default: 3.
        dpi: Resolution for matplotlib figures. Default: 80.
        facecolor: Background color for figures. Default: 'white'.
        font_path: Path to font for custom fonts. Default: None.
        ipython_format: IPython display format. Default: 'retina'.
        dpi_save: Resolution for saved figures. Default: 300.
        transparent: Whether to use transparent background. Default: None.
        scanpy: Whether to apply scanpy settings. Default: True.
        fontsize: Default font size for plots. Default: 14.
        color_map: Default color map for plots. Default: None.
        figsize: Default figure size. Default: None.
        vector_friendly: Control rasterization for vector-friendly plots. Default: True.
        
    Returns:
        None: The function configures global plotting settings and displays initialization information.
    """
    global _has_printed_logo
    import scanpy as sc

    console.level1(f"{EMOJI['start']} Starting plot initialization...")

    # 1) dependency check
    #print(f"{EMOJI['deps']} Checking dependencies...")
    #check_dependencies()
    # print(f"{EMOJI['done']} Dependencies OK")

    # 2) scanpy verbosity & figure params
    #print(f"{EMOJI['settings']} Applying plotting settings (verbosity={verbosity}, dpi={dpi})")
    console.node("Apply Scanpy/matplotlib settings", last=False, level=2)
    sc.settings.verbosity = verbosity
    import builtins
    is_ipython = getattr(builtins, "__IPYTHON__", False)
    if is_ipython:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        ipython_format = [ipython_format]
        set_matplotlib_formats(*ipython_format)
    
    from matplotlib import rcParams
    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if facecolor is not None:
        rcParams["figure.facecolor"] = facecolor
        rcParams["axes.facecolor"] = facecolor
    if scanpy:
        set_rcParams_scanpy(fontsize=fontsize, color_map=color_map)
    if figsize is not None:
        rcParams["figure.figsize"] = figsize
    
    # Set global vector_friendly setting
    global _vector_friendly
    _vector_friendly = vector_friendly
    #print(f"{EMOJI['done']} Settings applied")

    # 3) Custom font setup
    with console.group_node("Custom font setup", last=False, level=2):
        if font_path is not None:
            # Check if user wants Arial font (auto-download)
            if font_path.lower() in ['arial', 'arial.ttf'] and not font_path.endswith('.ttf'):
                try:
                    # Create a persistent cache location for the Arial font
                    import tempfile
                    import requests

                    cache_dir = tempfile.gettempdir()
                    cached_arial_path = os.path.join(cache_dir, 'omicverse_arial.ttf')
                    
                    # Check if Arial font is already cached
                    if os.path.exists(cached_arial_path):
                        console.level2(f"Using already downloaded Arial font from: {cached_arial_path}")
                        font_path = cached_arial_path
                    else:
                        console.level2("Downloading Arial font from GitHub...")
                        arial_url = "https://github.com/kavin808/arial.ttf/raw/refs/heads/master/arial.ttf"
                        
                        # Download the font
                        response = requests.get(arial_url, timeout=30)
                        response.raise_for_status()
                        
                        # Save the font to cache location
                        with open(cached_arial_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Use the cached font file
                        font_path = cached_arial_path
                        console.success(f"Arial font downloaded successfully to: {cached_arial_path}", level=2)
                    
                except Exception as e:
                    console.warn(f"Failed to download Arial font: {e}")
                    console.level2("Continuing with default font settings...")
                    font_path = None
        
        if font_path is not None:
            try:
                # 1) Create a brand-new manager
                fm.fontManager = fm.FontManager()
                
                # 2) Add your file
                fm.fontManager.addfont(font_path)
                
                # 3) Now find out what name it uses
                name = fm.FontProperties(fname=font_path).get_name()
                console.level2(f"Registered as: {name}")
                
                # 4) Point rcParams at that name
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                
            except Exception as e:
                console.warn(f"Failed to set custom font: {e}")
                console.level2("Continuing with default font settings...")

    # 4) suppress user/future/deprecation warnings
    #print(f"{EMOJI['warnings']} Suppressing common warnings")
    with console.group_node("Suppress warnings", last=False, level=2):
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
    #print(f"{EMOJI['done']} Warnings suppressed")

    # 6) print logo & version only once
    if not _has_printed_logo:
        #print(f"{EMOJI['logo']} OmicVerse Logo:")
        today = datetime.now()
        console.node(f"🔖 Version: {__version__}   📚 Tutorials: https://epione.readthedocs.io/", last=False, level=2)
        _has_printed_logo = True

    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = False

    console.success(f"{EMOJI['done']} plot_set complete.\n", level=1)


# Create aliases for backward compatibility
plotset = plot_set
ov_plot_set = plot_set


def set_rcParams_scanpy(fontsize=14, color_map=None):
    """Set matplotlib.rcParams to Scanpy defaults.

    Call this through :func:`scanpy.set_figure_params`.
    """
    # figure
    import matplotlib as mpl
    from cycler import cycler
    
    rcParams["figure.figsize"] = (4, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = 0.92 * fontsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = fontsize

    # legend
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycles
    rcParams["axes.prop_cycle"] = cycler(color=sc_color)

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize

    # axes grid
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = rcParams["image.cmap"] if color_map is None else color_map



sc_color=[
 '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', 
 '#FCBC10', '#EF7B77', '#279AD7','#F0EEF0',
 '#EAEFC5', '#7CBB5F','#368650','#A499CC','#5E4D9A',
 '#78C2ED','#866017', '#9F987F','#E0DFED',
 '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48',
 '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

red_color=['#F0C3C3','#E07370','#CB3E35','#A22E2A','#5A1713',
           '#D3396D','#8B0000', '#A52A2A', '#CD5C5C', '#DC143C' ]

green_color=['#91C79D','#8FC155','#56AB56','#2D5C33','#BBCD91',
             '#6E944A','#A5C953','#3B4A25','#010000']

orange_color=['#EFBD49','#D48F3E','#AC8A3E','#7D7237','#745228',
              '#E1C085','#CEBC49','#EBE3A1','#6C6331','#8C9A48','#D7DE61']

blue_color=['#1F577B', '#279AD7', '#78C2ED', '#01A0A7', '#75C8CC', '#9DC3C3',
            '#3E8CB1', '#52B3AD', '#265B58', '#5860A7', '#312C6C', '#4CC9F0']

purple_color=['#823d86','#825b94','#bb98c6','#c69bc6','#a69ac9',
              '#c5a6cc','#caadc4','#d1c3d4']

#more beautiful colors
# 28-color palettes with distinct neighboring colors
palette_28 = sc_color[:28]
