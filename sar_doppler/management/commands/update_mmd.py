"""Update MMD files"""
import pytz
import logging
import netCDF4
import datetime

import multiprocessing as mp

from dateutil.parser import parse

from django.db import connection
from django.db.utils import OperationalError

from django.utils import timezone
from django.contrib.gis.geos import WKTReader
from django.core.management.base import BaseCommand

from geospaas.utils.utils import nansat_filename
from geospaas.catalog.models import DatasetURI

from sar_doppler.utils import module_name, create_mmd_file
from sar_doppler.models import Dataset


def process(ds):

    def add_metadata_update_info(file, note="Correct geographic extent (rectangle)",
                                 type="Minor modification"):
        """ Add update information """
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            for line in lines:
                if note is not None and '</mmd:last_metadata_update>' in line:
                    f.write("    <mmd:update>\n"
                            "      <mmd:datetime>%s</mmd:datetime>\n"
                            "      <mmd:type>%s</mmd:type>\n"
                            "      <mmd:note>%s</mmd:note>\n"
                            "    </mmd:update>\n" % (datetime.datetime.utcnow().replace(
                                                     tzinfo=pytz.utc).strftime(
                                                         "%Y-%m-%dT%H:%M:%SZ"), type, note))
                f.write(line)
        # END function

    reprocess_mmd = False
    status = False
    db_locked = True
    while db_locked:
        try:
            uri = ds.dataseturi_set.get(uri__contains="ASA_WSD", uri__endswith=".nc")
        except OperationalError:
            db_locked = True
        except DatasetURI.DoesNotExist:
            return False
        else:
            db_locked = False
    connection.close()

    with netCDF4.Dataset(nansat_filename(uri.uri)) as ncds:
        if ncds.geospatial_lon_max > 180:
            reprocess_mmd = True

    if reprocess_mmd:
        db_locked = True
        while db_locked:
            try:
                new_uri, created = create_mmd_file(ds, uri)
            except OperationalError:
                db_locked = True
            else:
                db_locked = False
        connection.close()

        add_metadata_update_info(nansat_filename(new_uri.uri))
        status = True

    return status


class Command(BaseCommand):
    help = ("Post-processing of ingested GSAR RVL files and generation of png images for "
            "display in Leaflet")

    def add_arguments(self, parser):
        parser.add_argument("--file", type=str, default="")
        parser.add_argument("--polarisation", type=str)
        parser.add_argument("--start-date", type=str)
        parser.add_argument("--end-date", type=str)
        parser.add_argument("--log_file", type=str, default="update_mmd_files.log")

    def handle(self, *args, **options):

        logging.basicConfig(filename=options["log_file"], level=logging.INFO)

        tz = timezone.utc
        if options["start_date"]:
            start_date = tz.localize(parse(options["start_date"]))
        else:
            start_date = datetime.datetime(2002, 1, 1, tzinfo=timezone.utc)
        if options["end_date"]:
            end_date = tz.localize(parse(options["end_date"]))
        else:
            end_date = datetime.datetime.now(tz=timezone.utc)

        datasets = Dataset.objects.filter(
            time_coverage_start__range=[start_date, end_date],
            dataseturi__uri__endswith=".gsar").order_by("time_coverage_start")

        if options["file"]:
            datasets = datasets.filter(dataseturi__uri__contains=options["file"])

        if options["polarisation"]:
            datasets = datasets.filter(
                sardopplerextrametadata__polarization=options["polarisation"])

        num_unprocessed = len(datasets)

        logging.info("Checking MMD files of %d datasets" % num_unprocessed)
        # pool = mp.Pool(32)
        # res = pool.map(process, datasets)
        # logging.info("Successfully processed %d of %d MMD files." % (sum(bool(x) for x in res),
        #                                                              num_unprocessed))
        i = 0
        for ds in datasets:
            status = process(ds)
            if status:
                i += 1
        logging.info("Successfully processed (%d/%d)" % (i, num_unprocessed))
