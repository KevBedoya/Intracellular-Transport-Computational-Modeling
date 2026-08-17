from numba import njit
import math
import numpy as np
from auxiliary_tools import unit_conversion_functions as uni
import pandas as pd
import system_configuration as sys_config

'''
    Configuration of dependencies for computational_tools
'''

# storing dependency titles/aliases which can be accessed across this package
__all__ = ['math', 'njit', 'np', 'pd', 'uni', 'sys_config']
