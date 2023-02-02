from io import StringIO
from unittest.mock import patch, Mock, DEFAULT

from django.test import TestCase
from django.core.management import call_command

from sar_doppler.utils import *
from sar_doppler.models import Dataset
from sar_doppler.managers import DatasetManager

class TestProcessingSARDoppler(TestCase):

    fixtures = ["vocabularies"]

    #def test_process_sar_doppler(self):
    #    out = StringIO()
    #    wf = 'file://localhost/mnt/10.11.12.231/sat_auxdata/model/ncep/gfs/' \
    #            'gfs20091116/gfs.t18z.master.grbf03'
    #    call_command('ingest', wf, stdout=out)
    #    f = 'file://localhost/mnt/10.11.12.231/sat_downloads_asar/level-0/' \
    #            'gsar_rvl/RVL_ASA_WS_20091116195940116.gsar'
    #    call_command('ingest_sar_doppler', f, stdout=out)
    #    self.assertIn('Successfully added:', out.getvalue())

    #@patch.multiple(DatasetManager, filter=DEFAULT, process=DEFAULT,
    #    exclude=Mock(return_value=None))
    #def test_process_ingested_sar_doppler(self, filter, process):
    #    #mock_ds_objects.filter.return_value = mock_ds_objects
    #    #mock_ds_objects.exclude.return_value = mock_ds_objects
    #    #mock_ds_objects.process.return_value = (mock_ds_objects, True)

    #    out = StringIO()
    #    call_command('process_ingested_sar_doppler', stdout=out)
    #    filter.assert_called()
    #    #exclude.assert_called_once()
    #    process.assert_called_once()

    #def test_export(self):


class TestUtils(TestCase):

    fixtures = ["vocabularies", "sar_doppler"]

    def test_create_history_message(self):
        history_message = create_history_message(
                "sar_doppler.models.Dataset.objects.export2netcdf(n, ds, ",
                filename="some_filename.nc")
        self.assertIn(
            "sar_doppler.models.Dataset.objects.export2netcdf(n, ds, "
            "filename='some_filename.nc')",
            history_message)

        force = True
        args = [1,'hei']
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
        ds = Dataset.objects.get(pk=1)

        pp = path_to_nc_products(ds)
        self.assertEqual(pp, "/lustre/storeB/project/fou/fd/project/"
                             "sar-doppler/products/sar_doppler/2011/"
                             "01/04/RVL_ASA_WS_20110104102507222")

    def test_nc_name(self):
        ds = Dataset.objects.get(pk=1)
        ii = 2
        self.assertEqual(
            nc_name(ds, ii),
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/"
            "sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath2.nc")

    def test_path_to_nc_file(self):
        fn = "/path/should/be/removed/by/the/method/RVL_ASA_WS_20110104102507222subswath2.nc"
        ds = Dataset.objects.get(pk=1)
        self.assertEqual(
            path_to_nc_file(ds, fn),
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/"
            "sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath2.nc")

    def test_path_to_nc_file__old_is_correct(self):
        ds = Dataset.objects.get(pk=1)
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
        ds = Dataset.objects.get(pk=1)
        old_fn, new_fn = move_files_and_update_uris(ds, dry_run=False)
        self.assertEqual(old_fn[0],
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/sar_doppler/"
            "RVL_ASA_WS_20110104102507222/RVL_ASA_WS_20110104102507222subswath0.nc")
        self.assertEqual(new_fn[0],
            "/lustre/storeB/project/fou/fd/project/sar-doppler/products/sar_doppler/2011/01/04/"
            "RVL_ASA_WS_20110104102507222/RVL_ASA_WS_20110104102507222subswath0.nc")
        self.assertEqual(ds.dataseturi_set.get(uri__endswith="subswath0.nc").uri,
            "file://localhost/lustre/storeB/project/fou/fd/project/sar-doppler/products/"
            "sar_doppler/2011/01/04/RVL_ASA_WS_20110104102507222/"
            "RVL_ASA_WS_20110104102507222subswath0.nc")
        mock_os_rename.assert_called_with(old_fn[0], new_fn[0])

    @patch("sar_doppler.utils.os.rename")
    def test_move_files_and_update_uris__dry_run(self, mock_os_rename):
        mock_os_rename.return_value = None
        ds = Dataset.objects.get(pk=1)
        old_fn, new_fn = move_files_and_update_uris(ds, dry_run=True)
        self.assertEqual(ds.dataseturi_set.get(uri__endswith="subswath0.nc").uri,
            "file://localhost/lustre/storeB/project/fou/fd/project/sar-doppler/products/"
            "sar_doppler/RVL_ASA_WS_20110104102507222/"
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
        ds = Dataset.objects.get(pk=1)
        old_fn, new_fn = move_files_and_update_uris(ds, dry_run=True)
        self.assertEqual(old_fn, [])
        self.assertEqual(new_fn, [])
