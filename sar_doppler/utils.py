'''
Utility functions for processing Doppler from multiple SAR acquisitions
'''
import os, datetime
import numpy as np

from nansat.nsr import NSR
from nansat.domain import Domain
from nansat.nansat import Nansat

from django.utils import timezone


def nansumwrapper(a, **kwargs):
    mx = np.isnan(a).all(**kwargs)
    res = np.nansum(a, **kwargs)
    res[mx] = np.nan
    return res

def rb_model_func(x, a, b, c, d, e, f):
    return a + b*x + c*x**2 + d*x**3 + e*np.sin(x) + f*np.cos(x)
