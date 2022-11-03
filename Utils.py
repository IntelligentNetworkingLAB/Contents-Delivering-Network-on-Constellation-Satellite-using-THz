import scipy.integrate as sci
import numpy as np

def calc_propagtion_length_for_time_slot(sat_coverage, sat_velocity, airplane_velocity, alt_gap):
    total_length = 0
    relative_velocity = sat_velocity - airplane_velocity
    
    x = np.linspace(-sat_coverage/2, sat_coverage/2)
    y = alt_gap

    sci.

    return total_length
