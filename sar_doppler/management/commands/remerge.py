import logging

import multiprocessing as mp

from sar_doppler.models import Dataset
from sar_doppler.utils import create_mmd_file

from django.core.management.base import BaseCommand


def remerge(ds):
    m, uri = Dataset.objects.merge_swaths(ds)
    res = create_mmd_file(ds, uri)
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

