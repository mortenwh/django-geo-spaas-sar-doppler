""" Tests that don't require django""" - well, they do...
import unittest
from unittest.mock import patch

from sar_doppler.utils import *


class TestUtils(unittest.TestCase):

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

    @patch("sar_doppler.utils.nansat_filename")
    def test_path_to_nc_products(self, mock_nf):
        mock_nf.return_value = (
            "/lustre/storeB/project/fou/fd/project/sar-doppler/"
            "2011/01/04/RVL_ASA_WS_20110104102507222.gsar"
        )
        class Dataset:
            pass
        ds = Dataset()
        ds.time_coverage_start = datetime.datetime(2011, 1, 4, 10, 25, 7)
        #self.assertEqual(
        path_to_nc_products(ds)

    def test_path_to_nc_file(self):
        pass
