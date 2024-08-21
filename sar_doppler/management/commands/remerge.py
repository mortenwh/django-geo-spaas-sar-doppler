import os
import pytz
import logging
import netCDF4
import datetime

import multiprocessing as mp

from geospaas.utils.utils import nansat_filename

from sar_doppler.models import Dataset
from sar_doppler.utils import create_mmd_file

from django.db import connection
from django.db.utils import OperationalError
from django.core.management.base import BaseCommand


logging.basicConfig(filename='remerge_processed.log',
                    level=logging.INFO)


def try_creating_mmd(ds, nc_uri, mmd_uri0):
    try:
        mmd_uri, created = create_mmd_file(ds, nc_uri)
    except Exception as ee:
        logging.error(f"{ds.entry_id}: Failed to create MMD. {type(ee)}: {str(ee)}")
        return False
    if mmd_uri.uri != mmd_uri0.uri:
        mmd_uri0.delete()
    return created


def remerge(ds, force_nc=True):
    nc_created = False
    mmd_created = False
    locked = True
    while locked:
        try:
            mmd_uri0 = ds.dataseturi_set.get(uri__contains="ASA_WSD",
                                             uri__endswith=".xml")
        except OperationalError:
            locked = True
        else:
            locked = False
    connection.close()

    if os.path.isfile(nansat_filename(mmd_uri0.uri)) and not force_nc:
        logging.info(f"{ds.entry_id}: NC and MMD file already exists.")
        return nc_created, mmd_created

    locked = True
    while locked:
        try:
            nc_uri0 = ds.dataseturi_set.get(uri__contains="ASA_WSD",
                                            uri__endswith=".nc")
        except OperationalError:
            locked = True
        else:
            locked = False
    connection.close()

    xx = netCDF4.Dataset(nansat_filename(nc_uri0.uri))
    if datetime.datetime.fromisoformat(xx.date_created) > datetime.datetime(2024, 8, 21, 0, 0, 0,
                                                                            tzinfo=pytz.utc):
        return False, False

    if os.path.isfile(nansat_filename(nc_uri0.uri)) and not force_nc:
        logging.info(f"{ds.entry_id}: Merged file already created.")
        mmd_created = try_creating_mmd(ds, nc_uri0, mmd_uri0)
        if not mmd_created:
            logging.debug(f"{ds.entry_id}: MMD uri existed but not MMD file.")
    else:
        try:
            m, nc_uri = Dataset.objects.merge_swaths(ds)
        except Exception as ee:
            logging.error(f"{ds.entry_id}: Failed to create merged nc-file. {type(ee)}: {str(ee)}")
        else:
            nc_created = True
            if nc_uri.uri != nc_uri0.uri:
                nc_uri0.delete()
            mmd_created = try_creating_mmd(ds, nc_uri, mmd_uri0)

    return nc_created, mmd_created


class Command(BaseCommand):

    def handle(self, *args, **options):
        datasets = Dataset.objects.filter(dataseturi__uri__contains="ASA_WSD",
                                          dataseturi__uri__endswith=".nc")
        total = len(datasets)
        logging.info('Re-merging %d datasets' % total)
        pool = mp.Pool(32)
        res = pool.map(remerge, datasets)
        logging.info("Successfully re-merged %d of %d datasets." % (sum(bool(x[0]) for x in res),
                                                                    total))
        # for ds in datasets:
        #     nc_created, mmd_created = remerge(ds)
