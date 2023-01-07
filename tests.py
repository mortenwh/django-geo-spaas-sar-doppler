""" Tests that don't require django"""
import unittest

from sar_doppler.utils import create_history_message


class TestManagers(unittest.TestCase):

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
            
