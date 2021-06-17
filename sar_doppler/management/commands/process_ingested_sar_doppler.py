''' Processing of SAR Doppler from Norut's GSAR '''
import datetime
import logging

from dateutil.parser import parse

from django.utils import timezone
from django.core.management.base import BaseCommand

from nansat.exceptions import NansatGeolocationError

from sar_doppler.models import Dataset

logging.basicConfig(filename='process_ingested_sar_doppler.log', level=logging.INFO)

class Command(BaseCommand):
    help = 'Post-processing of ingested GSAR RVL files and generation of png images for ' \
            'display in Leaflet'

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default='')
        parser.add_argument('--lon-min', type=float, default=-180.0)
        parser.add_argument('--lon-max', type=float, default=180.0)
        parser.add_argument('--lat-min', type=float, default=-90.0)
        parser.add_argument('--lat-max', type=float, default=90.0)
        parser.add_argument('--polarisation', type=str)
        parser.add_argument('--start-date', type=str)
        parser.add_argument('--parallel', action='store_true')

    def handle(self, *args, **options):
        if options['start-date']:
            start_date = parse(options['start-date'], tzinfo=timezone.utc)
        else:
            start_date = datetime.datetime(2002,1,1, tzinfo=timezone.utc)
        if options['end-date']:
            end_date = parse(options['end-date'], tzinfo=timezone.utc)
        else:
            end_date = datetime.datetime.now(tz=timezone.utc)

        geometry = WKTReader().read(
                    "POLYGON ((%.1f %.1f, %.1f %.1f, %.1f %.1f, %.1f %.1f, %.1f %.1f))"
                    % (
                        options["lat-min"], options["lon-min"],
                        options["lat-max"], options["lon-min"],
                        options["lat-max", options["lon-max"],
                        options["lat-min"], options["lon-max"]
                    )
                )

        datasets = Dataset.objects.filter(
                time_coverage_start__range = [start_date, end_date],
                geographic_location__geometry__intersects = geometry,
        	dataseturi__uri__endswith='.gsar'
            ).order_by('time_coverage_start')

        if options['file']:
                datasets = datasets.filter(dataseturi__uri__contains = options['file']

        if options['polarisation']:
                datasets = datasets.filter(sardopplerextrametadata__polarization = 
                                              options['polarisation'])

        num_unprocessed = len(datasets)

        print('Processing %d datasets' %num_unprocessed)
        #if options['parallel']:

        #else:
        for i,ds in enumerate(unprocessed):
            uri = ds.dataseturi_set.get(uri__endswith='.gsar').uri
            try:
                updated_ds, processed = Dataset.objects.process(uri)
            except (ValueError, IOError, NansatGeolocationError):
                # some files manually moved to *.error...
                continue
            if processed:
                self.stdout.write('Successfully processed (%d/%d): %s\n' % (i+1, num_unprocessed,
                    uri))
            else:
                msg = 'Corrupt file (%d/%d, may have been partly processed): %s\n' %(i+1,
                    num_unprocessed, uri)
                logging.info(msg)
                self.stdout.write(msg)

