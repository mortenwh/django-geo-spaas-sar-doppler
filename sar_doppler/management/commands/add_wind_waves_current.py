""" Post-processing of SAR Doppler to add wind, waves and current """
from sar_doppler.management.commands.base import BaseSARDopplerCommand


class Command(BaseSARDopplerCommand):
    help = "Post-post-processing of Doppler files to get wind, waves and current"
    manager_method = "add_wind_waves_current"
