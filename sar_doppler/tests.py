import datetime

import numpy as np

from io import StringIO
from unittest.mock import patch, MagicMock

from django.test import TestCase
from django.core.management import call_command

from sar_doppler.utils import create_mmd_file
from sar_doppler.utils import path_to_nc_file
from sar_doppler.utils import path_to_nc_products
from sar_doppler.utils import module_name
from sar_doppler.utils import create_history_message
from sar_doppler.utils import nc_name
from sar_doppler.utils import move_files_and_update_uris
from sar_doppler.utils import reprocess_if_exported_before
from sar_doppler.models import Dataset

# Fixture dataset pk=1 has three URIs:
#   pk=1  file://localhost/.../RVL_ASA_WS_20110104102507222.gsar
#   pk=2  file://localhost/.../RVL_ASA_WS_20110104102507222subswath0.nc
#   pk=3  file://localhost/.../ASA_WSDV2PRNMI20110104_102507_...0000.nc  (contains "ASA_WSD")
FIXTURE_GSAR_URI = ("file://localhost/lustre/storeB/project/fou/fd/project/"
                    "sar-doppler/2011/01/04/RVL_ASA_WS_20110104102507222.gsar")


class TestProcessingSARDoppler(TestCase):

    fixtures = ["vocabularies", "sar_doppler"]

    def test_get_or_create(self):
        fn = ("file://localhost/lustre/storeB/project/fou/fd/project/sar-doppler/2008"
              "/01/01/RVL_ASA_WS_20080101230010941.gsar")
        ds, cr = Dataset.objects.get_or_create(fn)
        self.assertTrue(cr)

    def test_ingest_sar_doppler(self):
        fn = ("/lustre/storeB/project/fou/fd/project/sar-doppler/2005/03/08/"
              "RVL_ASA_WS_20050308225807190.gsar")
        out = StringIO()
        call_command('ingest_sar_doppler', fn, stdout=out)
        self.assertIn('Successfully added:', out.getvalue())

    def test_ingest_sar_doppler_wrong_path(self):
        fn = ("/lustre/storeB/project/fou/fd/project/sar-doppler/2012/01/14/"
              "RVL_ASA_WS_20120120114202288.gsar")
        out = StringIO()
        with self.assertLogs() as cm:
            call_command('ingest_sar_doppler', fn, stdout=out)
            self.assertIn("GSAR file is misplaced", cm.output[0])

    # def test_process_sar_doppler(self):
    #     out = StringIO()
    #     wf = 'file://localhost/mnt/10.11.12.231/sat_auxdata/model/ncep/gfs/' \
    #             'gfs20091116/gfs.t18z.master.grbf03'
    #     call_command('ingest', wf, stdout=out)
    #     f = 'file://localhost/mnt/10.11.12.231/sat_downloads_asar/level-0/' \
    #             'gsar_rvl/RVL_ASA_WS_20091116195940116.gsar'
    #     call_command('ingest_sar_doppler', f, stdout=out)
    #     self.assertIn('Successfully added:', out.getvalue())

    # @patch.multiple(DatasetManager, filter=DEFAULT, process=DEFAULT,
    #     exclude=Mock(return_value=None))
    # def test_process_ingested_sar_doppler(self, filter, process):
    #     #mock_ds_objects.filter.return_value = mock_ds_objects
    #     #mock_ds_objects.exclude.return_value = mock_ds_objects
    #     #mock_ds_objects.process.return_value = (mock_ds_objects, True)

    #     out = StringIO()
    #     call_command('process_ingested_sar_doppler', stdout=out)
    #     filter.assert_called()
    #     #exclude.assert_called_once()
    #     process.assert_called_once()

    # def test_export(self):


class TestUtils(TestCase):

    fixtures = ["vocabularies", "sar_doppler"]

    def setUp(self):
        self.ds = Dataset.objects.get(pk=1)

    def test_create_history_message(self):
        history_message = create_history_message(
            "sar_doppler.models.Dataset.objects.export2netcdf(n, ds, ",
            filename="some_filename.nc")
        self.assertIn(
            "sar_doppler.models.Dataset.objects.export2netcdf(n, ds, "
            "filename='some_filename.nc')",
            history_message)

        force = True
        args = [1, 'hei']
        kwargs = {'test': 'dette', 'test2': 1}
        history_message = create_history_message(
            "sar_doppler.models.Dataset.objects.process(ds, ",
            force=force, *args, **kwargs)
        self.assertIn(
            "sar_doppler.models.Dataset.objects.process(ds, 1, 'hei', force=True, test='dette', "
            "test2=1)", history_message)

    def test_module_name(self):
        self.assertEqual(module_name(), 'sar_doppler')

    def test_path_to_nc_products(self):
        ds = self.ds

        pp = path_to_nc_products(ds)
        self.assertEqual(pp, "/lustre/storeB/project/fou/fd/project/"
                             "sar-doppler/products/sar_doppler/2011/"
                             "01/04/RVL_ASA_WS_20110104102507222")

    def test_nc_name(self):
        ds = self.ds
        ii = 2
        self.assertEqual(
            nc_name(ds, ii),
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/"
            "sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath2.nc")

    def test_path_to_nc_file(self):
        fn = "/path/should/be/removed/by/the/method/RVL_ASA_WS_20110104102507222subswath2.nc"
        ds = self.ds
        self.assertEqual(
            path_to_nc_file(ds, fn),
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/"
            "sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath2.nc")

    def test_path_to_nc_file__old_is_correct(self):
        ds = self.ds
        fn = (
            "/lustre/storeB/project/fou/fd/project/sar-doppler/"
            "products/sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath0.nc")
        self.assertEqual(
            path_to_nc_file(ds, fn),
            "/lustre/storeB/project/fou/fd/project/sar-doppler/"
            "products/sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath0.nc")

    @patch("sar_doppler.utils.os.rename")
    def test_move_files_and_update_uris(self, mock_os_rename):
        mock_os_rename.return_value = None
        ds = self.ds
        old_fn, new_fn = move_files_and_update_uris(ds, dry_run=False)
        self.assertEqual(old_fn[0],
                         "/lustre/storeB/project/fou/fd/project/sar-doppler/products/sar_doppler/"
                         "RVL_ASA_WS_20110104102507222/RVL_ASA_WS_20110104102507222subswath0.nc")
        self.assertEqual(new_fn[0],
                         "/lustre/storeB/project/fou/fd/project/sar-doppler/products/sar_doppler/"
                         "2011/01/04/"
                         "RVL_ASA_WS_20110104102507222/RVL_ASA_WS_20110104102507222subswath0.nc")
        self.assertEqual(ds.dataseturi_set.get(uri__endswith="subswath0.nc").uri,
                         "file://localhost/lustre/storeB/project/fou/fd/project/sar-doppler/"
                         "products/sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
                         "RVL_ASA_WS_20110104102507222subswath0.nc")
        mock_os_rename.assert_called_with(old_fn[0], new_fn[0])

    @patch("sar_doppler.utils.os.rename")
    def test_move_files_and_update_uris__dry_run(self, mock_os_rename):
        mock_os_rename.return_value = None
        ds = self.ds
        old_fn, new_fn = move_files_and_update_uris(ds, dry_run=True)
        self.assertEqual(ds.dataseturi_set.get(uri__endswith="subswath0.nc").uri,
                         "file://localhost/lustre/storeB/project/fou/fd/project/sar-doppler/"
                         "products/sar_doppler/RVL_ASA_WS_20110104102507222/"
                         "RVL_ASA_WS_20110104102507222subswath0.nc")
        mock_os_rename.assert_not_called()

    @patch("sar_doppler.utils.product_path")
    @patch("sar_doppler.utils.nansat_filename")
    def test_move_files_and_update_uris__dry_run__old_is_same_as_new(self,
                                                                     mock_nansat_filename,
                                                                     mock_product_path):
        mock_nansat_filename.return_value = (
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/sar_doppler/2011/01/04/"
            "RVL_ASA_WS_20110104102507222/RVL_ASA_WS_20110104102507222subswath0.nc")
        mock_product_path.return_value = (
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/sar_doppler/2011/01/04/"
            "RVL_ASA_WS_20110104102507222")
        ds = self.ds
        old_fn, new_fn = move_files_and_update_uris(ds, dry_run=True)
        self.assertEqual(old_fn, [])
        self.assertEqual(new_fn, [])

    @patch("sar_doppler.models.Dataset.objects.process")
    @patch("sar_doppler.managers.os.path.getmtime")
    def test_reprocess_if_exported_before(self, mock_getmtime, mock_process):
        """ Test that a dataset is reprocessed if one of the nc files
        was processed before the given date, and not if after.
        """
        mock_getmtime.return_value = datetime.datetime(2023, 1, 20, 12, 0, 0)
        ds = self.ds
        mock_process.return_value = (ds, True)
        ds, proc = reprocess_if_exported_before(ds, datetime.datetime(2023, 1, 21))
        self.assertTrue(proc)
        ds, proc = reprocess_if_exported_before(ds, datetime.datetime(2023, 1, 19))
        self.assertFalse(proc)

    @patch("sar_doppler.utils.nc_to_mmd.Nc_to_mmd")
    @patch("sar_doppler.utils.product_path")
    @patch("sar_doppler.utils.DatasetURI.objects.get_or_create")
    def test_create_mmd_file(self, mock_goc, mock_pp, mock_Nc2mmd):
        """Test creation of MMD file.
        """
        mmd_uri = ("file://localhost/lustre/storeB/project/fou/fd/project"
                   "/sar-doppler/products/sar_doppler/mmd/2011/01/04/"
                   "ASA_WSDV2PRNMI20110104_102507_000609353111_00396_52134_0000.xml")

        class MockURI:
            uri = mmd_uri

        class Mock_Nc_to_mmd:
            def to_mmd(*a, **k):
                return (True, "")

        mock_Nc2mmd.return_value = Mock_Nc_to_mmd()

        ds = self.ds
        uri = ds.dataseturi_set.get(uri__contains="ASA_WSD")
        mock_pp.return_value = ("/lustre/storeB/project/fou/fd/project/sar-doppler/"
                                "products/sar_doppler/mmd/2011/01/04/")
        mock_goc.return_value = (MockURI(), True)
        new_uri, created = create_mmd_file(ds, uri, check_only=False)
        self.assertEqual(new_uri.uri, mmd_uri)
        self.assertTrue(created)


class TestDatasetManagerProcess(TestCase):
    """Unit tests for DatasetManager.process() — the paths we intend to refactor."""

    fixtures = ["vocabularies", "sar_doppler"]

    def setUp(self):
        self.ds = Dataset.objects.get(pk=1)

    @patch("sar_doppler.managers.create_mmd_file")
    def test_already_processed_returns_false(self, mock_create_mmd):
        """Returns (ds, False) when an ASA_WSD .nc URI already exists."""
        mock_create_mmd.return_value = (MagicMock(), True)
        _, status = Dataset.objects.process(self.ds)
        self.assertFalse(status)

    @patch("sar_doppler.managers.create_mmd_file")
    def test_already_processed_checks_for_mmd(self, mock_create_mmd):
        """When already processed and no .xml MMD exists, create_mmd_file is called."""
        mock_create_mmd.return_value = (MagicMock(), True)
        Dataset.objects.process(self.ds)
        mock_create_mmd.assert_called_once()

    @patch("sar_doppler.managers.Doppler")
    def test_all_subswaths_fail_raises_runtime_error(self, mock_Doppler):
        """Raises RuntimeError when all 5 subswaths fail to load."""
        mock_Doppler.side_effect = Exception("GSAR read error")
        # Remove the "already processed" .nc so the method attempts fresh processing.
        self.ds.dataseturi_set.filter(uri__contains="ASA_WSD", uri__endswith=".nc").delete()
        with self.assertRaises(RuntimeError):
            Dataset.objects.process(self.ds)

    @patch("sar_doppler.managers.Doppler")
    def test_some_subswaths_fail_raises_runtime_error(self, mock_Doppler):
        """Raises RuntimeError when some (not all) subswaths fail."""
        good = MagicMock()
        good.__getitem__ = MagicMock(return_value=np.zeros((10, 10)))
        # First subswath fails, the rest succeed.
        mock_Doppler.side_effect = [Exception("subswath 0 error"), good, good, good, good]
        self.ds.dataseturi_set.filter(uri__contains="ASA_WSD", uri__endswith=".nc").delete()
        with self.assertRaises(RuntimeError):
            Dataset.objects.process(self.ds)


class TestWorkerProcess(TestCase):
    """Unit tests for the management command base.process() worker function."""

    def _mock_ds(self, uri=FIXTURE_GSAR_URI):
        """Return a mock dataset whose .gsar URI lookup returns *uri*."""
        mock_uri = MagicMock()
        mock_uri.uri = uri
        mock_ds = MagicMock()
        mock_ds.dataseturi_set.get.return_value = mock_uri
        return mock_ds

    @patch("sar_doppler.management.commands.base.Dataset")
    def test_success_returns_true_and_uri(self, mock_Dataset):
        """Returns (True, uri) when the manager method succeeds."""
        mock_ds = self._mock_ds()
        mock_Dataset.objects.my_method.return_value = (mock_ds, True)

        from sar_doppler.management.commands.base import process
        status, uri = process(mock_ds, "my_method")

        self.assertTrue(status)
        self.assertEqual(uri, FIXTURE_GSAR_URI)

    @patch("sar_doppler.management.commands.base.Dataset")
    def test_already_done_returns_false_not_none(self, mock_Dataset):
        """Returns (False, uri) — not (None, uri) — when manager returns False."""
        mock_ds = self._mock_ds()
        mock_Dataset.objects.my_method.return_value = (mock_ds, False)

        from sar_doppler.management.commands.base import process
        status, uri = process(mock_ds, "my_method")

        self.assertFalse(status)
        self.assertIsNotNone(status)  # False, not None — must not appear in unprocessed list

    @patch("sar_doppler.management.commands.base.Dataset")
    def test_exception_returns_none(self, mock_Dataset):
        """Returns (None, uri) when the manager method raises any exception."""
        mock_ds = self._mock_ds()
        mock_Dataset.objects.my_method.side_effect = RuntimeError("processing failed")

        from sar_doppler.management.commands.base import process
        status, uri = process(mock_ds, "my_method")

        self.assertIsNone(status)
        self.assertEqual(uri, FIXTURE_GSAR_URI)

    @patch("sar_doppler.management.commands.base.Dataset")
    def test_db_retry_on_operational_error(self, mock_Dataset):
        """Retries the .gsar URI lookup when OperationalError is raised."""
        from django.db.utils import OperationalError
        mock_uri = MagicMock()
        mock_uri.uri = FIXTURE_GSAR_URI
        mock_ds = MagicMock()
        # First call raises OperationalError, second succeeds.
        mock_ds.dataseturi_set.get.side_effect = [OperationalError("locked"), mock_uri]
        mock_Dataset.objects.my_method.return_value = (mock_ds, True)

        from sar_doppler.management.commands.base import process
        status, uri = process(mock_ds, "my_method")

        self.assertEqual(mock_ds.dataseturi_set.get.call_count, 2)
        self.assertTrue(status)
