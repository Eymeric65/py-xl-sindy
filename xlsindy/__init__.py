import importlib.metadata

__version__ = importlib.metadata.version("py-xl-sindy")

from . import catalog_gen
from . import dynamics_modeling
from . import euler_lagrange
from . import optimization
from . import render
from . import simulation
from . import result_formatting


