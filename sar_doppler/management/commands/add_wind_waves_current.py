import fcntl
import logging
import datetime
import time

import multiprocessing as mp

from dateutil.parser import parse

from django.db import connection
from django.db.utils import OperationalError

from django.utils import timezone
from django.contrib.gis.geos import WKTReader
from django.core.management.base import BaseCommand

from sar_doppler.models import Dataset

FORMATTER = logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s")

_log_file = None


class FlockFileHandler(logging.FileHandler):
    """Process-safe log handler using fcntl.flock().

    Unlike threading.Lock-based handlers, flock is automatically released
    by the OS if the process dies, preventing deadlocks in multiprocessing.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            with open(self.baseFilename, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(msg + self.terminator)
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            self.handleError(record)


def worker_init(log_file):
    global _log_file
    _log_file = log_file
    handler = FlockFileHandler(log_file)
    handler.setFormatter(FORMATTER)
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def process(ds):
    status = False
    uri = "(unknown)"
    db_locked = True
    while db_locked:
        try:
            uri = ds.dataseturi_set.get(uri__endswith=".gsar").uri
        except OperationalError:
            time.sleep(1)
        else:
            db_locked = False
    connection.close()
    try:
        updated_ds, status = Dataset.objects.add_wind_waves_current(ds)
    except Exception as e:
        logging.error("%s: %s (%s)" % (type(e), str(e), uri))

    return status


class Command(BaseCommand):
    help = ("Post-post-processing of Doppler files to get wind, waves"
            "and current")

    def add_arguments(self, parser):
        parser.add_argument("--file", type=str, default="")
        parser.add_argument("--wkt", type=str,
                            default="POLYGON ((-180 -90, -180 90, 180 90, 180 -90, -180 -90))")
        parser.add_argument("--polarisation", type=str)
        parser.add_argument("--start-date", type=str)
        parser.add_argument("--end-date", type=str)
        parser.add_argument("--wind", type=str)
        parser.add_argument("--log_file", type=str, default="process_ingested_sar_doppler.log")

    def handle(self, *args, **options):

        file_handler = FlockFileHandler(options["log_file"])
        file_handler.setFormatter(FORMATTER)
        root = logging.getLogger()
        root.handlers = []
        root.addHandler(file_handler)
        root.setLevel(logging.INFO)

        tz = timezone.utc
        if options["start_date"]:
            start_date = tz.localize(parse(options["start_date"]))
        else:
            start_date = datetime.datetime(2002, 1, 1, tzinfo=timezone.utc)
        if options["end_date"]:
            end_date = tz.localize(parse(options["end_date"]))
        else:
            end_date = datetime.datetime.now(tz=timezone.utc)

        geometry = WKTReader().read(options["wkt"])

        datasets = Dataset.objects.filter(
            time_coverage_start__range=[start_date, end_date],
            geographic_location__geometry__intersects=geometry,
            dataseturi__uri__endswith=".gsar").order_by("time_coverage_start")

        if options["file"]:
            datasets = datasets.filter(dataseturi__uri__contains=options["file"])

        if options["polarisation"]:
            datasets = datasets.filter(
                sardopplerextrametadata__polarization=options["polarisation"])

        num_unprocessed = len(datasets)

        logging.info("Processing %d datasets" % num_unprocessed)
        pool = mp.Pool(32, initializer=worker_init, initargs=(options["log_file"],))
        try:
            res = pool.map(process, datasets)
        except Exception as e:
            logging.error("pool.map failed: %s" % str(e))
            res = []
        finally:
            pool.close()
            pool.join()

        logging.info("Successfully processed %d of %d datasets." % (sum(bool(x) for x in res),
                                                                    num_unprocessed))
        # i = 0
        # for ds in datasets:
        #     status = process(ds)
        #     if status:
        #         i += 1
        # logging.info("Successfully processed (%d/%d)" % (i, num_unprocessed))

        processed = Dataset.objects.filter(
            time_coverage_start__range=[start_date, end_date],
            geographic_location__geometry__intersects=geometry,
            dataseturi__uri__contains="ASA_WSD",
            dataseturi__uri__endswith=".nc",
            sardopplerextrametadata__polarization=options["polarisation"])
        if options["file"]:
            processed = processed.filter(dataseturi__uri__contains=options["file"])
        logging.info(f"In total, {len(processed)} of {num_unprocessed} "
                     "nc files have been processed.")
        processed = Dataset.objects.filter(
            time_coverage_start__range=[start_date, end_date],
            geographic_location__geometry__intersects=geometry,
            dataseturi__uri__contains="ASA_WSD",
            dataseturi__uri__endswith=".xml",
            sardopplerextrametadata__polarization=options["polarisation"])
        logging.info(f"In total, {len(processed)} of {num_unprocessed} "
                     "xml files have been processed.")

        file_handler.close()

