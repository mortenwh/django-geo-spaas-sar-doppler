import sys
import tqdm
import cmocean
import logging

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from nansat.nsr import NSR
from nansat.domain import Domain
from nansat.nansat import Nansat

from geospaas.utils.utils import nansat_filename

from sar_doppler.models import Dataset

from django.core.management.base import BaseCommand


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_reprojected(url, dom, satpass, *args, **kwargs):
    try:
        w = Nansat(url)
    except RuntimeError as ee:
        logging.error(f"{url} failing: {str(ee)}")
        return None
    odir = w.get_metadata()["orbit_direction"]
    if satpass is not None and odir != satpass:
        return None
    w.reproject(dom, *args, **kwargs)
    return w


def valid_ocean_data(datasets, dom, satpass=None):
    """Sum of valid Doppler pixels over sea areas.
    """
    count = 0
    for i in tqdm.tqdm(range(len(datasets))):
        ds = datasets[i]
        fn = nansat_filename(ds.dataseturi_set.get(uri__contains="ASA_WSD",
                                                   uri__endswith=".nc").uri)
        w = get_reprojected(fn, dom, satpass, resample_alg=0, tps=True, block_size=5)
        if w is None:
            continue
        valid = w["valid_sea_doppler"]
        fdg = w["fdg"]
        valid[fdg > 100] = 0
        valid[fdg < -100] = 0
        valid[np.isnan(valid)] = 0
        if count == 0:
            vv = valid.copy()
            lon, lat = w.get_geolocation_grids()
        else:
            vv1 = valid.copy()
            vv0 = vv.copy()
            vv = None
            vv = vv0 + vv1
            vv0 = None
            vv1 = None

        count += 1
        valid = None
        fdg = None
        w = None
        if count > 10:
            break

    return lon, lat, vv


class Command(BaseCommand):

    def handle(self, *args, **options):
        datasets = Dataset.objects.filter(dataseturi__uri__contains="ASA_WSD",
                                          dataseturi__uri__endswith=".nc")
        total = len(datasets)
        logging.info('Counting coverage in %d datasets' % total)
        lonmin, lonmax, latmin, latmax = -179, 179, -90, 90
        dom = Domain(NSR().wkt, f"-te {lonmin} {latmin} {lonmax} {latmax} -tr 0.1 0.1")
        lon, lat, valid = valid_ocean_data(datasets, dom)
        da = xr.DataArray(valid, dims=["y", "x"], coords={"lat": (("y", "x"), lat),
                                                          "lon": (("y", "x"), lon)})
        da.to_netcdf("data_coverage.nc")
        #ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        #da.plot.pcolormesh("lon", "lat", ax=ax1, cmap=cmocean.cm.amp, add_colorbar=True)
        #ax1.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
        #ax1.gridlines(draw_labels=True)
        #plt.show()
        # Funker ikke:
        # plt.savefig("data_coverage.eps", papertype="a4", transparent=True, orientation="portrait")
