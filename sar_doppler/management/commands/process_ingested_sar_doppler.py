''' Processing of SAR Doppler from Norut's GSAR '''
import logging
import datetime

import multiprocessing as mp

from dateutil.parser import parse

from django.db import connection
from django.db.utils import OperationalError

from django.utils import timezone
from django.contrib.gis.geos import WKTReader
from django.core.management.base import BaseCommand

from sar_doppler.models import Dataset

logging.basicConfig(filename='process_ingested_sar_doppler.log',
                    level=logging.INFO)


def process(ds):
    status = False
    db_locked = True
    while db_locked:
        try:
            uri = ds.dataseturi_set.get(uri__endswith='.gsar').uri
        except OperationalError:
            db_locked = True
        else:
            db_locked = False
    connection.close()
    try:
        updated_ds, status = Dataset.objects.process(ds)
    except Exception as e:
        logging.error("%s: %s (%s)" % (type(e), str(e), uri))
    return status


class Command(BaseCommand):
    help = ("Post-processing of ingested GSAR RVL files and generation of png images for "
            "display in Leaflet")

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default='')
        parser.add_argument('--wkt', type=str,
                            default='POLYGON ((-180 -90, -180 90, 180 90, 180 -90, -180 -90))')
        # parser.add_argument('--lon-min', type=float, default=-180.0)
        # parser.add_argument('--lon-max', type=float, default=180.0)
        # parser.add_argument('--lat-min', type=float, default=-90.0)
        # parser.add_argument('--lat-max', type=float, default=90.0)
        parser.add_argument('--polarisation', type=str)
        parser.add_argument('--start-date', type=str)
        parser.add_argument('--end-date', type=str)
        parser.add_argument('--wind', type=str)

    def handle(self, *args, **options):
        tz = timezone.utc
        if options['start_date']:
            start_date = tz.localize(parse(options['start_date']))
        else:
            start_date = datetime.datetime(2002, 1, 1, tzinfo=timezone.utc)
        if options['end_date']:
            end_date = tz.localize(parse(options['end_date']))
        else:
            end_date = datetime.datetime.now(tz=timezone.utc)

        geometry = WKTReader().read(options['wkt'])

        datasets = Dataset.objects.filter(
            time_coverage_start__range=[start_date, end_date],
            geographic_location__geometry__intersects=geometry,
            dataseturi__uri__endswith='.gsar').order_by('time_coverage_start')

        if options['file']:
            datasets = datasets.filter(dataseturi__uri__contains=options['file'])

        if options['polarisation']:
            datasets = datasets.filter(
                sardopplerextrametadata__polarization=options['polarisation'])

        num_unprocessed = len(datasets)

        logging.info('Processing %d datasets' % num_unprocessed)
        pool = mp.Pool(32)
        res = pool.map(process, datasets)
        logging.info("Successfully processed %d of %d datasets." % (sum(bool(x) for x in res),
                                                                    num_unprocessed))
        # i = 0
        # for ds in datasets:
        #     status = process(ds)
        #     i += 1
        # logging.info('Successfully processed (%d/%d)' % (i, num_unprocessed))
