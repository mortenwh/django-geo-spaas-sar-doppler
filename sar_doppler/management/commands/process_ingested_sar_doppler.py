''' Processing of SAR Doppler from Norut's GSAR '''
import sys
import datetime
import logging
import traceback

import multiprocessing as mp
import parmap

from dateutil.parser import parse

from django.utils import timezone
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import WKTReader

from nansat.exceptions import NansatGeolocationError

from geospaas.utils.utils import nansat_filename

from sardoppler.utils import AttitudeFileError
from sardoppler.sardoppler import FixThisError, AttitudeError
from sar_doppler.models import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
lfile = logging.FileHandler("process_ingested_sar_doppler.log", encoding="utf-8")
logger.addHandler(lfile)

#logging.basicConfig(filename='process_ingested_sar_doppler.log', encoding='utf-8',
#                    level=logging.ERROR)

class Command(BaseCommand):
    help = 'Post-processing of ingested GSAR RVL files and generation of png images for ' \
            'display in Leaflet'

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default='')
        parser.add_argument('--wkt', type=str, 
                            default='POLYGON ((-180 -90, -180 90, 180 90, 180 -90, -180 -90))')
        #parser.add_argument('--lon-min', type=float, default=-180.0)
        #parser.add_argument('--lon-max', type=float, default=180.0)
        #parser.add_argument('--lat-min', type=float, default=-90.0)
        #parser.add_argument('--lat-max', type=float, default=90.0)
        parser.add_argument('--polarisation', type=str)
        parser.add_argument('--start-date', type=str)
        parser.add_argument('--end-date', type=str)
        parser.add_argument('--wind', type=str)

    def handle(self, *args, **options):
        tz = timezone.utc
        if options['start_date']:
            start_date = tz.localize(parse(options['start_date']))
        else:
            start_date = datetime.datetime(2002,1,1, tzinfo=timezone.utc)
        if options['end_date']:
            end_date = tz.localize(parse(options['end_date']))
        else:
            end_date = datetime.datetime.now(tz=timezone.utc)

        geometry = WKTReader().read(options['wkt'])

        datasets = Dataset.objects.filter(
                time_coverage_start__range = [start_date, end_date],
                geographic_location__geometry__intersects = geometry,
        	dataseturi__uri__endswith='.gsar'
            ).order_by('time_coverage_start')

        if options['file']:
            datasets = datasets.filter(dataseturi__uri__contains = options['file'])

        if options['polarisation']:
            datasets = datasets.filter(sardopplerextrametadata__polarization = 
                                          options['polarisation'])

        num_unprocessed = len(datasets)

        i = 0
        logging.info('Processing %d datasets' %num_unprocessed)
        for ds in datasets:
            status = self.process(ds, options['wind'])
            uri = ds.dataseturi_set.get(uri__endswith='.gsar')
            i += 1
            if status:
                self.stdout.write('Successfully processed (%d/%d): %s\n' % (
                    i, num_unprocessed, uri.uri))
                logger.info('Successfully processed (%d/%d): %s' % (
                    i, num_unprocessed, uri.uri))
            else:
                logger.info('%s was already processed (%d/%d)' % (
                    uri.uri, i, num_unprocessed))
            #i = self.process_and_log(ds, options['wind'], i)
        # This is failing:
        #pool = mp.Pool(mp.cpu_count())
        #res = pool.map(self.process_and_log, datasets, options['wind'])
        #parmap.map(self.process_and_log, datasets, options['wind']) #, pm_bar=False)

    def process_and_log(self, ds, wind, i):
        status = self.process(ds, wind)
        if not status:
            return None
        i += 1
        uri = ds.dataseturi_set.get(uri__endswith='.gsar')
        self.stdout.write('Successfully processed (%d/%d): %s\n' % (
                i, num_unprocessed, uri.uri))
        logger.info('%s' % nansat_filename(uri.uri))
        return i

    def process(self, ds, wind):
        status = False
        uri = ds.dataseturi_set.get(uri__endswith='.gsar').uri
        try:
            updated_ds, status = Dataset.objects.process(ds, wind=wind)
        except (RuntimeError, AttitudeError, AttitudeFileError, EOFError, FixThisError) as e:
            # some files also manually moved to *.error...
            einfo = sys.exc_info()
            logger.error("%s: %s" % (einfo[1], uri))
            logger.error(traceback.format_exc())
        return status
