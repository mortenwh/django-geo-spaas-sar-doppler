import logging
import netCDF4

import multiprocessing as mp

from geospaas.utils.utils import nansat_filename

from sar_doppler.models import Dataset
from sar_doppler.utils import create_mmd_file

from django.core.management.base import BaseCommand


logging.basicConfig(filename='remerge_processed.log',
                    level=logging.INFO)


def remerge(ds):
    uri = ds.dataseturi_set.get(uri__contains="ASA_WSD",
                                uri__endswith=".nc")
    nc = netCDF4.Dataset(nansat_filename(uri.uri))
    if "orbit_direction" in nc.ncattrs():
        logging.info("NC-file already updated")
        return False
    m, uri = Dataset.objects.merge_swaths(ds)
    try:
        res = create_mmd_file(ds, uri)
    except:
        logging.error(f"{uri} failed")
        raise
    return res[1]

class Command(BaseCommand):

    def handle(self, *args, **options):
        datasets = Dataset.objects.filter(dataseturi__uri__contains="ASA_WSD",
                                          dataseturi__uri__endswith=".nc")
        total = len(datasets)
        logging.info('Re-merging %d datasets' % total)
        pool = mp.Pool(32)
        res = pool.map(remerge, datasets)
        logging.info("Successfully re-merged %d of %d datasets." % (sum(bool(x) for x in res),
                                                                    total))

