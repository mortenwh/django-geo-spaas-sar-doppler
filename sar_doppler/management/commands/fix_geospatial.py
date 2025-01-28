import os
import glob
import logging
import netCDF4
import datetime

from multiprocessing import Pool

from nansat.nansat import Nansat

from django.db import connection
from django.db.utils import OperationalError
from django.core.management.base import BaseCommand

from geospaas.utils.utils import nansat_filename

from sar_doppler.utils import create_mmd_file

from sar_doppler.models import Dataset


def update_nc_metadata(file):
    try:
        ds = Dataset.objects.get(dataseturi__uri__contains=file)
    except:
        return
    nc_uri = ds.dataseturi_set.get(uri__contains="ASA_WSD", uri__endswith=".nc")
    ff = netCDF4.Dataset(nansat_filename(nc_uri.uri), "a")
    lon = ff["longitude"][:].filled()
    lat = ff["latitude"][:].filled()
    ff.geospatial_lat_max = lat.max()
    ff.geospatial_lat_min = lat.min()
    ff.geospatial_lon_max = lon.max()
    ff.geospatial_lon_min = lon.min()
    ff.close()
    db_locked = True
    while db_locked:
        try:
            create_mmd_file(ds, nc_uri)
        except OperationalError:
            db_locked = True
        else:
            db_locked = False
    connection.close()


def find_nc_files_created_before(time="2025-01-27T14:50:57.445429",
                                 folder="/lustre/storeB/project/fou/fd/project/sar-doppler"
                                        "/products/sar_doppler/merged"):
    files = glob.glob(os.path.join(folder, "**/*.nc"), recursive=True)
    created_before = []
    for file in files:
        created = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        if created < datetime.datetime.fromisoformat(time):
            created_before.append(file)
    return created_before


def update_all_nc_files_before(time="2025-01-27T14:50:57.445429"):
    files = find_nc_files_created_before(time=time)
    logging.info('Fixing geolocation of %d datasets' % len(files))
    pool = Pool(processes=32)
    # for file in files:
    #     update_nc_metadata(file)
    pool.map(update_nc_metadata, files)


class Command(BaseCommand):

    def handle(self, *args, **options):
        update_all_nc_files_before()
