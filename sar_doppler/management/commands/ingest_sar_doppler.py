''' Ingestion of Doppler products from Norut's GSAR '''
import logging
from optparse import make_option

from nansat.exceptions import NansatGeolocationError

from django.core.exceptions import MultipleObjectsReturned
from django.db.utils import IntegrityError
from django.core.management.base import BaseCommand

from geospaas.utils.utils import uris_from_args
from geospaas.catalog.models import DatasetURI
from geospaas.catalog.models import Dataset as catalogDataset

from sar_doppler.models import Dataset
from sar_doppler.errors import AlreadyExists
import os

class Command(BaseCommand):
    args = '<filename>'
    help = 'Add WS file to catalog archive and make png images for ' \
            'display in Leaflet'

    def add_arguments(self, parser):
        parser.add_argument('gsar_files', nargs='*', type=str)
        parser.add_argument('--logfile', action='store_true', help='Logfilename')
        parser.add_argument('--reprocess', action='store_true', 
                help='Force reprocessing')

    def handle(self, *args, **options):

        logfilename = options.pop('logfile', 'ingest_sar_doppler.log')
        logging.basicConfig(filename=logfilename, level=logging.CRITICAL)

        for uri in uris_from_args(options['gsar_files']):
            self.stdout.write('Ingesting %s ...\n' % uri)
            try:
                ds, cr = Dataset.objects.get_or_create(uri, **options)
            except Exception as e:
                logging.critical(uri+': '+repr(e))
                continue
            if not type(ds)==catalogDataset:
                self.stdout.write('Not found: %s\n' % uri)
            elif cr:
                self.stdout.write('Successfully added: %s\n' % uri)
            else:
                if not type(ds) == catalogDataset:
                    self.stdout.write('Not found: %s\n' % uri)
                elif cr:
                    self.stdout.write('Successfully added: %s\n' % uri)
                else:
                    self.stdout.write('Was already added: %s\n' % uri)



