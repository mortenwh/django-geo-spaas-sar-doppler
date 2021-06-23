''' Processing of SAR Doppler from Norut's GSAR '''
import datetime
import logging
import parmap

from dateutil.parser import parse

from django.utils import timezone
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import WKTReader

from nansat.exceptions import NansatGeolocationError

from geospaas.utils.utils import nansat_filename

from sardoppler.utils import FixThisError
from sar_doppler.models import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
lfile = logging.FileHandler("process_ingested_sar_doppler.log", encoding="utf-8")
logger.addHandler(lfile)

#logging.basicConfig(filename='process_ingested_sar_doppler.log', encoding='utf-8',
#                    level=logging.ERROR)

def process(ds):
    status = False
    uri = ds.dataseturi_set.get(uri__endswith='.gsar').uri
    try:
        updated_ds, processed = Dataset.objects.process(ds)
    except Exception as e:
        # some files manually moved to *.error...
        logger.error("%s: %s" % (str(e), uri))
    else:
        status = True
    return status

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
        parser.add_argument('--parallel', action='store_true')

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
        print('Processing %d datasets' %num_unprocessed)
        if options['parallel']:
            parmap.map(process, datasets, pm_pbar=True)
        else:
            for ds in datasets:
                status = process(ds)
                if not status:
                    continue
                i += 1
                self.stdout.write('Successfully processed (%d/%d): %s\n' % (
                        i+1, num_unprocessed, ds.dataseturi_set.get(uri__endswith='.gsar').uri))
                for uri in ds.dataseturi_set.filter(uri__endswith='.nc'):
                    logger.info('%s' % nansat_filename(uri.uri))
