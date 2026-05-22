""" Processing of SAR Doppler from Norut's GSAR """
from sar_doppler.management.commands.base import BaseSARDopplerCommand


class Command(BaseSARDopplerCommand):
    help = ("Post-processing of ingested GSAR RVL files and generation of png images for "
            "display in Leaflet")
    manager_method = "process"
