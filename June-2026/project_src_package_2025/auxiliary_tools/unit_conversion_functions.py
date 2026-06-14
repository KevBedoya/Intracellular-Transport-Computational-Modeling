from . import math
from numba import njit


@njit(nopython=True, cache=True)
def convert_bytes(num_bytes):
    """
    Converts num_bytes to the largest possible unit conversion of bytes

    :param num_bytes: (float) number of bytes
    :return: (str) the associated conversion to some unit of bytes
    """
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB"]
    factor = 1024.0
    for unit in units:
        if num_bytes < factor:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= factor
    return f"{num_bytes:.2f} QB"


def convert_seconds(seconds):
    """

    Converts seconds to the largest possible unit conversion of second

    :param seconds: (float) number of seconds
    :return: (str) the associated conversion to some unit of time
    """

    string_units = None
    if seconds < 1:
        if 10**(-12) < seconds <= 10**(-8):
            seconds *= 10**9
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + " nanosecond"
            else:
                return string_units + " nanoseconds"
        elif 10**(-8) < seconds <= 10**(-5):
            seconds *= 10**6
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + " microsecond"
            else:
                return string_units + " microseconds"
        elif 10**(-5) < seconds < 1:
            seconds *= 10**3
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + " millisecond"
            else:
                return string_units + " milliseconds"
    elif 1 <= seconds < 60:
        string_units = f'{seconds}'
        if seconds == 1:
            return string_units + " second"
        else:
            return string_units + " seconds"
    elif seconds >= 60:
        if 60 <= seconds < 3600:
            seconds /= 60
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + ' minute'
            else:
                return string_units + ' minutes'
        elif 3600 <= seconds < 3600 * 24:
            seconds /= 3600
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + ' hour'
            else:
                return string_units + ' hours'
        elif 24 * 3600 <= seconds < 168 * 3600:
            seconds /= (3600 * 24)
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + ' day'
            else:
                return string_units + ' days'
        elif 168 * 3600 <= seconds <= math.inf:
            seconds /= (3600 * 168)
            string_units = f'{seconds}'
            if seconds == 1:
                return string_units + ' week'
            else:
                return string_units + ' weeks'


def simulation_time_conversion(rg_param, ry_param, time_steps):
    """
    Converts a given number of time-steps from a biophysical computation to its associated
    dimensionless time. (This time is referred to as "sim" or "simulation" time across the entire project)

    :param rg_param: (int) # of radial rings in the cellular domain
    :param ry_param: (int) # of angular rays in the cellular domain
    :param time_steps: (int) # of discrete time steps
    :return: simulation time
    """
    d_radius = 1 / rg_param
    d_theta = ((2 * math.pi) / ry_param)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / 2
    return time_steps * d_time




