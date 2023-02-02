'''
Utility functions for processing Doppler from multiple SAR acquisitions
'''
import os, datetime
import numpy as np
import logging

from nansat.nsr import NSR
from nansat.domain import Domain
from nansat.nansat import Nansat

from django.conf import settings
from django.utils import timezone
from django.db import connection

from geospaas.utils.utils import nansat_filename
from geospaas.utils.utils import product_path

def nansumwrapper(a, **kwargs):
    mx = np.isnan(a).all(**kwargs)
    res = np.nansum(a, **kwargs)
    res[mx] = np.nan
    return res

def rb_model_func(x, a, b, c, d, e, f):
    return a + b*x + c*x**2 + d*x**3 + e*np.sin(x) + f*np.cos(x)

def create_history_message(caller, *args, **kwargs):
    history_message = "%s: %s" % (datetime.datetime.now(timezone.utc).isoformat(), caller)
    if bool(args):
        for arg in args:
            if type(arg)==str:
                history_message += "\'%s\', " % arg
            else:
                history_message += "%s, " % str(arg)
    if bool(kwargs):
        for key in kwargs.keys():
            if kwargs[key]:
                if type(kwargs[key])==str:
                    history_message += "%s=\'%s\', " % (key, kwargs[key])
                else:
                    history_message += "%s=%s, " % (key, str(kwargs[key]))
    return history_message[:-2] + ")"

def module_name():
    """ Get module name
    """
    return __name__.split('.')[0]

def path_to_nc_products(ds):
    """ Get the (product) path to netCDF-CF files."""
    pp = product_path(module_name(),
                      nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri),
                      date=ds.time_coverage_start)
    connection.close()
    return pp

def path_to_nc_file(ds, fn):
    """ Get the path to a netcdf product with filename fn."""
    return os.path.join(path_to_nc_products(ds), os.path.basename(fn))

def nc_name(ds, ii):
    """ Get the full path filename of exported netcdf of subswath ii."""
    fn = path_to_nc_file(ds, os.path.basename(nansat_filename(
        ds.dataseturi_set.get(uri__endswith='.gsar').uri)).split('.')[0] +
        'subswath%s.nc' % ii)
    connection.close()
    return fn

def move_files_and_update_uris(ds, dry_run=True):
    """ Get the uris of the netcdf products of a dataset, get the
    new ones (with yyyy/mm/dd/), move the files to the new
    location, and update the uris."""
    old, new = [], []
    for uri in ds.dataseturi_set.filter(uri__endswith=".nc",
                                        uri__contains=settings.PRODUCTS_ROOT):
        old_fn = nansat_filename(uri.uri)
        new_fn = path_to_nc_file(ds, nansat_filename(uri.uri))
        logging.info("Move %s ---> %s" % (old_fn, new_fn))
        if not dry_run:
            uri.uri = "file://localhost" + new_fn
            uri.save()
            os.rename(old_fn, new_fn)
            assert nansat_filename(uri.uri) == new_fn
        else:
            logging.info("Dry-run....")
        connection.close()
        old.append(old_fn)
        new.append(new_fn)
    return old, new
