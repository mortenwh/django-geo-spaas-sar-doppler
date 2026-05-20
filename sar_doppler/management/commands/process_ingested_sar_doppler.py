""" Processing of SAR Doppler from Norut's GSAR """
import logging
import logging.handlers
import datetime
import signal
import time
import threading

import multiprocessing as mp
from multiprocessing.pool import WorkerLostError

from dateutil.parser import parse

from django.db import connection
from django.db.utils import OperationalError

from django.utils import timezone
from django.contrib.gis.geos import WKTReader
from django.core.management.base import BaseCommand

from sar_doppler.models import Dataset
from sar_doppler.management.commands import worker_init

FORMATTER = logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s")

TASK_TIMEOUT = 600  # seconds — kills task if wind_waves_doppler (GDAL) hangs


def _handle_timeout(signum, frame):
    raise TimeoutError("Task timed out after %d seconds" % TASK_TIMEOUT)


def process(ds):
    """Process one dataset."""
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

    stop_heartbeat = threading.Event()
    start_time = time.monotonic()

    def _heartbeat():
        while not stop_heartbeat.wait(60):
            elapsed = int(time.monotonic() - start_time)
            logging.info("Heartbeat: still processing %s (%ds elapsed)" % (uri, elapsed))

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(TASK_TIMEOUT)
    try:
        updated_ds, status = Dataset.objects.process(ds)
    except TimeoutError as e:
        logging.error("TIMEOUT %s (%s)" % (str(e), uri))
    except Exception as e:
        logging.error("%s: %s (%s)" % (type(e), str(e), uri))
    finally:
        signal.alarm(0)
        stop_heartbeat.set()

    return status, uri


class Command(BaseCommand):
    help = ("Post-processing of ingested GSAR RVL files and generation of png images for "
            "display in Leaflet")

    def add_arguments(self, parser):
        parser.add_argument("--file", type=str, default="")
        parser.add_argument("--wkt", type=str,
                            default="POLYGON ((-180 -90, -180 90, 180 90, 180 -90, -180 -90))")
        # parser.add_argument("--lon-min", type=float, default=-180.0)
        # parser.add_argument("--lon-max", type=float, default=180.0)
        # parser.add_argument("--lat-min", type=float, default=-90.0)
        # parser.add_argument("--lat-max", type=float, default=90.0)
        parser.add_argument("--polarisation", type=str)
        parser.add_argument("--start-date", type=str)
        parser.add_argument("--end-date", type=str)
        parser.add_argument("--wind", type=str)
        parser.add_argument("--log_file", type=str, default="process_ingested_sar_doppler.log")

    def handle(self, *args, **options):

        file_handler = logging.FileHandler(options["log_file"])
        file_handler.setFormatter(FORMATTER)
        log_queue = mp.Queue(-1)
        listener = logging.handlers.QueueListener(log_queue, file_handler,
                                                  respect_handler_level=True)
        listener.start()

        root = logging.getLogger()
        root.handlers = []
        root.addHandler(logging.handlers.QueueHandler(log_queue))
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
        pool = mp.Pool(32, initializer=worker_init, initargs=(log_queue,))
        try:
            res = pool.map(process, datasets)
        except WorkerLostError as e:
            logging.error("Worker crashed: %s" % str(e))
            res = []
        except Exception as e:
            logging.error("pool.map failed: %s" % str(e))
            res = []
        finally:
            pool.close()
            pool.join()
            listener.stop()

        logging.info("Successfully processed %d of %d datasets." % (
            sum(bool(s) for s, _ in res), num_unprocessed))
        failed = [uri for s, uri in res if not s]
        if failed:
            logging.info("Unprocessed datasets (%d):" % len(failed))
            for uri in failed:
                logging.info("  %s" % uri)
        # i = 0
        # for ds in datasets:
        #     status = process(ds)
        #     i += 1
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
