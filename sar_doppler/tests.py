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

    def test_path_to_nc_file(self):
        pass
