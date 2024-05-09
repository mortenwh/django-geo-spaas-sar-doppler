'''
Utility functions for processing Doppler from multiple SAR acquisitions
'''
import datetime
import logging
import netCDF4
import os
import pathlib

import numpy as np

from py_mmd_tools import nc_to_mmd

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


def lut_results_path(lutfilename):
    """ Get the base product path for files resulting from a specific LUT."""
    return product_path(module_name(), lutfilename)


class MockDataset:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def create_mmd_files(lutfilename, nc_uris):
    """Create MMD files for the provided dataset nc uris."""
    base_url = "https://thredds.met.no/thredds/dodsC/remotesensingenvisat/asar-doppler"
    dataset_citation = {
        "author": "Morten W. Hansen, Jeong-Won Park, Geir Engen, Harald Johnsen",
        "publication_date": "2023-10-05",
        "title": "Calibrated geophysical ENVISAT ASAR wide-swath range Doppler frequency shift",
        "publisher": "European Space Agency (ESA), Norwegian Meteorological Institute (MET Norway)",
        "doi": "https://doi.org/10.57780/esa-56fb232"
    }
    for uri in nc_uris:
        url = base_url + uri.uri.split('sar_doppler')[-1]
        outfile = os.path.join(
            lut_results_path(lutfilename),
            pathlib.Path(pathlib.Path(nansat_filename(uri.uri)).stem).with_suffix('.xml')
        )
        logging.info("Creating MMD file: %s" % outfile)
        md = nc_to_mmd.Nc_to_mmd(nansat_filename(uri.uri), opendap_url=url,
                                 output_file=outfile)
        ds = netCDF4.Dataset(nansat_filename(uri.uri))
        dataset_citation['url'] = "https://data.met.no/dataset/%s" % ds.id
        ds.close()
        req_ok, msg = md.to_mmd(dataset_citation=dataset_citation)
    return req_ok, msg


def move_files_and_update_uris(ds, dry_run=True):
    """ Get the uris of the netcdf products of a gsar rvl dataset,
    get the new ones (with yyyy/mm/dd/), move the files to the new
    location, and update the uris."""
    old, new = [], []
    if bool(ds.dataseturi_set.filter(uri__endswith=".gsar")):
        for uri in ds.dataseturi_set.filter(uri__endswith=".nc",
                                            uri__contains=settings.PRODUCTS_ROOT):
            old_fn = nansat_filename(uri.uri)
            new_fn = path_to_nc_file(ds, nansat_filename(uri.uri))
            if old_fn==new_fn:
                continue
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

def reprocess_if_exported_before(ds, date_before):
    """ Reprocess datasets that were last processed before a given
    date.
    """
    from sar_doppler.models import Dataset
    nc_uris = ds.dataseturi_set.filter(uri__contains='.nc')
    if nc_uris:
        nc_uris = nc_uris.filter(uri__contains='subswath')
    reprocess = False
    proc = False
    for uri in nc_uris:
        if datetime.datetime.fromtimestamp(
                os.path.getmtime(nansat_filename(uri.uri))) < date_before:
            reprocess = True
    if reprocess:
        ds, proc = Dataset.objects.process(ds, force=True)
    else:
        logging.info("Already reprocessed: %s" % os.path.basename(nansat_filename(uri.uri)).split('.')[0])
    return ds, proc


def create_merged_swaths(ds, EPSG = 4326, **kwargs):
    """Merge swaths, add dataseturi, and return Nansat object.

    EPSG options:
        - 4326: WGS 84 / longlat
        - 3995: WGS 84 / Arctic Polar Stereographic
    """
    nn = {}
    nn[0] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc' % 0).uri))
    lon0, lat0 = nn[0].get_geolocation_grids()
    nn[1] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc' % 1).uri))
    lon1, lat1 = nn[1].get_geolocation_grids()
    nn[2] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc' % 2).uri))
    lon2, lat2 = nn[2].get_geolocation_grids()
    nn[3] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc' % 3).uri))
    lon3, lat3 = nn[3].get_geolocation_grids()
    nn[4] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc' % 4).uri))
    lon4, lat4 = nn[4].get_geolocation_grids()

    gg = gsar(nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri))

    connection.close()

    i0 = gg.getinfo(channel=0)
    i1 = gg.getinfo(channel=1)
    i2 = gg.getinfo(channel=2)
    i3 = gg.getinfo(channel=3)
    i4 = gg.getinfo(channel=4)

    i0_ytimes = np.arange(0, i0.ysize/i0.ysamplefreq, 1/i0.ysamplefreq)
    i1_ytimes = np.arange(0, i1.ysize/i1.ysamplefreq, 1/i1.ysamplefreq)
    i2_ytimes = np.arange(0, i2.ysize/i2.ysamplefreq, 1/i2.ysamplefreq)
    i3_ytimes = np.arange(0, i3.ysize/i3.ysamplefreq, 1/i3.ysamplefreq)
    i4_ytimes = np.arange(0, i4.ysize/i4.ysamplefreq, 1/i4.ysamplefreq)

    i0_abstimesy = []
    for i in range(i0.ysize):
        i0_abstimesy.append(i0.ytime.dtime + timedelta(seconds=i0_ytimes[i]))

    i1_abstimesy = []
    for i in range(i1.ysize):
        i1_abstimesy.append(i1.ytime.dtime + timedelta(seconds=i1_ytimes[i]))

    i2_abstimesy = []
    for i in range(i2.ysize):
        i2_abstimesy.append(i2.ytime.dtime + timedelta(seconds=i2_ytimes[i]))

    i3_abstimesy = []
    for i in range(i3.ysize):
        i3_abstimesy.append(i3.ytime.dtime + timedelta(seconds=i3_ytimes[i]))

    i4_abstimesy = []
    for i in range(i4.ysize):
        i4_abstimesy.append(i4.ytime.dtime + timedelta(seconds=i4_ytimes[i]))
    
    t0 = np.min(np.array([i0.ytime.dtime, i1.ytime.dtime, i2.ytime.dtime, i3.ytime.dtime,
                          i4.ytime.dtime]))
    
    i0_dt = np.array(i0_abstimesy) - t0
    i1_dt = np.array(i1_abstimesy) - t0
    i2_dt = np.array(i2_abstimesy) - t0
    i3_dt = np.array(i3_abstimesy) - t0
    i4_dt = np.array(i4_abstimesy) - t0
    
    # Determine line numbers
    i0_N = []
    for i in range(i0.ysize):
        i0_N.append(np.floor((i0_dt[i].seconds+i0_dt[i].microseconds*10**(-6))*i0.ysamplefreq))
        
    i1_N = []
    for i in range(i1.ysize):
        i1_N.append(np.floor((i1_dt[i].seconds+i1_dt[i].microseconds*10**(-6))*i1.ysamplefreq))
        
    i1_N = []
    for i in range(i1.ysize):
        i1_N.append(np.floor((i1_dt[i].seconds+i1_dt[i].microseconds*10**(-6))*i1.ysamplefreq))
        
    i2_N = []
    for i in range(i2.ysize):
        i2_N.append(np.floor((i2_dt[i].seconds+i2_dt[i].microseconds*10**(-6))*i2.ysamplefreq))
        
    i3_N = []
    for i in range(i3.ysize):
        i3_N.append(np.floor((i3_dt[i].seconds+i3_dt[i].microseconds*10**(-6))*i3.ysamplefreq))
        
    i4_N = []
    for i in range(i4.ysize):
        i4_N.append(np.floor((i4_dt[i].seconds+i4_dt[i].microseconds*10**(-6))*i4.ysamplefreq))
    
