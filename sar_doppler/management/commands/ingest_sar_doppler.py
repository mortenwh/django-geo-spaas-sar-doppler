""" Ingestion of Doppler products from Norut's GSAR """
import os
import logging
import sys

from django.core.management.base import BaseCommand

from django.db.utils import OperationalError
from django.db import connection

from geospaas.utils.utils import uris_from_args
from geospaas.utils.utils import nansat_filename
from geospaas.catalog.models import Dataset as catalogDataset

from sar_doppler.models import Dataset


def ingest(uri):
    created = 0
    fn = nansat_filename(uri)
    dir_date_str = os.path.dirname(fn)[-10:].replace("/", "")
    file_date_str = os.path.basename(fn)[11:19]
    if dir_date_str != file_date_str and dir_date_str[2:] != file_date_str[:-2]:
        logging.error("GSAR file is misplaced: %s" % fn)
        return 0
    logging.debug('Ingesting %s ...\n' % uri)
    retry = True
    while retry:
        try:
            ds, cr = Dataset.objects.get_or_create(uri)
        except OperationalError as oe:
            logging.debug(str(oe) + " - retrying %s" % uri)
            retry = True
        except Exception as e:
            logging.error(uri+': '+repr(e))
            retry = False
            return 0
        else:
            retry = False
        connection.close()
    if not type(ds) == catalogDataset:
        logging.error('Failed to create: %s\n' % uri)
    elif cr:
        logging.debug('Successfully added: %s\n' % uri)
        created += 1
    else:
        logging.debug('Was already added: %s\n' % uri)

    return created


class Command(BaseCommand):
    args = '<filename>'
    help = ("Add WS file to catalog archive and make png images for "
            "display in Leaflet")

    def add_arguments(self, parser):
        parser.add_argument('gsar_files', nargs='*', type=str)
        parser.add_argument('--logfile', action='store_true', help='Logfilename')
        parser.add_argument('--log-to-stdout', action='store_true')

    def handle(self, *args, **options):

        if not options['log_to_stdout']:
            if not options['logfile']:
                logfilename = 'ingest_sar_doppler.log'
            else:
                logfilename = options['logfile']
            logging.basicConfig(filename=logfilename, level=logging.INFO)
        else:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        created = 0
        uris = uris_from_args(options['gsar_files'])
        for uri in uris:
            created += ingest(uri)
        logging.info("Added %d/%d datasets" % (created, len(uris)))

        # pool = mp.Pool(32)
        # created = pool.map(ingest, uris)
        # logging.info("Added %d/%d datasets" % (sum(created), len(uris)))
