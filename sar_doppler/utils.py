'''
Utility functions for processing Doppler from multiple SAR acquisitions
'''
import os
import csv
import pytz
import logging
import netCDF4
import pathlib
import datetime

import numpy as np

from scipy.interpolate import CubicSpline

from py_mmd_tools import nc_to_mmd

import pythesint as pti

from nansat.domain import Domain
from nansat.nansat import Nansat

from django.conf import settings
from django.utils import timezone
from django.db import connection

from sardoppler.gsar import gsar
from sardoppler.sardoppler import Doppler

from geospaas.utils.utils import nansat_filename
from geospaas.utils.utils import product_path


def nansumwrapper(a, **kwargs):
    mx = np.isnan(a).all(**kwargs)
    res = np.nansum(a, **kwargs)
    res[mx] = np.nan
    return res


def rb_model_func(x, a, b, c, d, e, f):
    return a + b*x + c*x**2 + d*x**3 + e*np.sin(x) + f*np.cos(x)


def create_history_message(caller, *args, **kwargs):
    history_message = "%s: %s" % (datetime.datetime.now(timezone.utc).isoformat(), caller)
    if bool(args):
        for arg in args:
            if type(arg) == str:
                history_message += "\'%s\', " % arg
            else:
                history_message += "%s, " % str(arg)
    if bool(kwargs):
        for key in kwargs.keys():
            if kwargs[key]:
                if type(kwargs[key]) == str:
                    history_message += "%s=\'%s\', " % (key, kwargs[key])
                else:
                    history_message += "%s=%s, " % (key, str(kwargs[key]))
    return history_message[:-2] + ")"


def module_name():
    """ Get module name
    """
    return __name__.split('.')[0]


def path_to_merged(ds, fn):
    """Get the path to a merged file with basename fn.
    """
    pp = product_path(module_name() + ".merged", "", date=ds.time_coverage_start)
    connection.close()
    return os.path.join(pp, os.path.basename(fn))


def path_to_nc_products(ds):
    """ Get the (product) path to netCDF-CF files."""
    pp = product_path(module_name(),
                      nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri),
                      date=ds.time_coverage_start)
    connection.close()
    return pp


def path_to_nc_file(ds, fn):
    """ Get the path to a netcdf product with filename fn."""
    return os.path.join(path_to_nc_products(ds), os.path.basename(fn))


def nc_name(ds, ii):
    """ Get the full path filename of exported netcdf of subswath ii."""
    fn = path_to_nc_file(ds, os.path.basename(nansat_filename(
        ds.dataseturi_set.get(uri__endswith='.gsar').uri)).split('.')[0] + 'subswath%s.nc' % ii)
    connection.close()
    return fn


def lut_results_path(lutfilename):
    """ Get the base product path for files resulting from a specific LUT."""
    return product_path(module_name(), lutfilename)


class MockDataset:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def create_mmd_file(lutfilename, uri):
    """Create MMD files for the provided dataset nc uris."""
    base_url = "https://thredds.met.no/thredds/dodsC/remotesensingenvisat/asar-doppler"
    dataset_citation = {
        "author": "Morten W. Hansen, Jeong-Won Park, Geir Engen, Harald Johnsen",
        "publication_date": "2023-10-05",
        "title": "Calibrated geophysical ENVISAT ASAR wide-swath range Doppler frequency shift",
        "publisher":
            "European Space Agency (ESA), Norwegian Meteorological Institute (MET Norway)",
        "doi": "https://doi.org/10.57780/esa-56fb232"
    }
    url = base_url + uri.uri.split('sar_doppler/merged')[-1]
    outfile = os.path.join(
        lut_results_path(lutfilename),
        pathlib.Path(pathlib.Path(nansat_filename(uri.uri)).stem).with_suffix('.xml')
    )

    wms_base_url = "https://fastapi.s-enda.k8s.met.no/api/get_quicklook"

    path_parts = nansat_filename(uri.uri).split("/")
    year = path_parts[-4]
    month = path_parts[-3]
    day = path_parts[-2]
    wms_url = os.path.join(wms_base_url, year, month, day, path_parts[-1])
    layers = ["fdg", "incidence_angle", "topographic_height", "valid_doppler", "fe", "fgeo", "fww",
              "wind_direction", "wind_speed"]

    logging.info("Creating MMD file: %s" % outfile)
    md = nc_to_mmd.Nc_to_mmd(nansat_filename(uri.uri), opendap_url=url,
                             output_file=outfile)
    ds = netCDF4.Dataset(nansat_filename(uri.uri))
    dataset_citation['url'] = "https://data.met.no/dataset/%s" % ds.id
    ds.close()
    req_ok, msg = md.to_mmd(dataset_citation=dataset_citation, checksum_calculation=True,
                            parent="no.met:e19b9c36-a9dc-4e13-8827-c998b9045b54",
                            add_wms_data_access=True, wms_link=wms_url, wms_layer_names=layers)
    return req_ok, msg


def move_files_and_update_uris(ds, dry_run=True):
    """ Get the uris of the netcdf products of a gsar rvl dataset,
    get the new ones (with yyyy/mm/dd/), move the files to the new
    location, and update the uris."""
    old, new = [], []
    if bool(ds.dataseturi_set.filter(uri__endswith=".gsar")):
        for uri in ds.dataseturi_set.filter(uri__endswith=".nc",
                                            uri__contains=settings.PRODUCTS_ROOT):
            old_fn = nansat_filename(uri.uri)
            new_fn = path_to_nc_file(ds, nansat_filename(uri.uri))
            if old_fn == new_fn:
                continue
            logging.info("Move %s ---> %s" % (old_fn, new_fn))
            if not dry_run:
                uri.uri = "file://localhost" + new_fn
                uri.save()
                os.rename(old_fn, new_fn)
                assert nansat_filename(uri.uri) == new_fn
            else:
                logging.info("Dry-run....")
            connection.close()
            old.append(old_fn)
            new.append(new_fn)
    return old, new


def reprocess_if_exported_before(ds, date_before):
    """ Reprocess datasets that were last processed before a given
    date.
    """
    from sar_doppler.models import Dataset
    nc_uris = ds.dataseturi_set.filter(uri__contains='.nc')
    if nc_uris:
        nc_uris = nc_uris.filter(uri__contains='subswath')
    reprocess = False
    proc = False
    for uri in nc_uris:
        if datetime.datetime.fromtimestamp(
                os.path.getmtime(nansat_filename(uri.uri))) < date_before:
            reprocess = True
    if reprocess:
        ds, proc = Dataset.objects.process(ds, force=True)
    else:
        logging.info("Already reprocessed: %s" % os.path.basename(
            nansat_filename(uri.uri)).split('.')[0])
    return ds, proc


def create_merged_swaths(ds, EPSG=4326, **kwargs):
    """Merge swaths, add dataseturi, and return Nansat object.

    EPSG options:
        - 4326: WGS 84 / longlat
        - 3995: WGS 84 / Arctic Polar Stereographic
    """
    nn = {}
    nn[0] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='swath%d.nc' % 0).uri))
    lon0, lat0 = nn[0].get_geolocation_grids()
    nn[1] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='swath%d.nc' % 1).uri))
    lon1, lat1 = nn[1].get_geolocation_grids()
    nn[2] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='swath%d.nc' % 2).uri))
    lon2, lat2 = nn[2].get_geolocation_grids()
    nn[3] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='swath%d.nc' % 3).uri))
    lon3, lat3 = nn[3].get_geolocation_grids()
    nn[4] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='swath%d.nc' % 4).uri))
    lon4, lat4 = nn[4].get_geolocation_grids()

    gg = gsar(nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri))

    connection.close()

    i0 = gg.getinfo(channel=0)
    i1 = gg.getinfo(channel=1)
    i2 = gg.getinfo(channel=2)
    i3 = gg.getinfo(channel=3)
    i4 = gg.getinfo(channel=4)

    i0_ytimes = np.arange(0, i0.ysize/i0.ysamplefreq, 1/i0.ysamplefreq)
    i1_ytimes = np.arange(0, i1.ysize/i1.ysamplefreq, 1/i1.ysamplefreq)
    i2_ytimes = np.arange(0, i2.ysize/i2.ysamplefreq, 1/i2.ysamplefreq)
    i3_ytimes = np.arange(0, i3.ysize/i3.ysamplefreq, 1/i3.ysamplefreq)
    i4_ytimes = np.arange(0, i4.ysize/i4.ysamplefreq, 1/i4.ysamplefreq)

    t0 = np.min(np.array([i0.ytime.dtime, i1.ytime.dtime, i2.ytime.dtime, i3.ytime.dtime,
                          i4.ytime.dtime]))

    i0_dt = i0_ytimes + (i0.ytime.dtime-t0).total_seconds()
    i1_dt = i1_ytimes + (i1.ytime.dtime-t0).total_seconds()
    i2_dt = i2_ytimes + (i2.ytime.dtime-t0).total_seconds()
    i3_dt = i3_ytimes + (i3.ytime.dtime-t0).total_seconds()
    i4_dt = i4_ytimes + (i4.ytime.dtime-t0).total_seconds()

    lon0, lat0 = nn[0].get_geolocation_grids()
    lon1, lat1 = nn[1].get_geolocation_grids()
    lon2, lat2 = nn[2].get_geolocation_grids()
    lon3, lat3 = nn[3].get_geolocation_grids()
    lon4, lat4 = nn[4].get_geolocation_grids()

    def resample_azim(xnew, x, y):
        yy = np.interp(xnew, x, y, left=np.nan, right=np.nan)
        xs = xnew[~np.isnan(yy)]
        ys = yy[~np.isnan(yy)]
        z = CubicSpline(xs, ys, bc_type='natural')
        ynew = z(xnew, nu=0)
        return ynew

    # Interpolate lon/lat
    # subswath 0
    lon0i = np.empty((lon2.shape[0], lon0.shape[1]))
    lat0i = np.empty((lat2.shape[0], lat0.shape[1]))
    for i in range(lon0.shape[1]):
        lon0i[:, i] = resample_azim(i2_dt, i0_dt, lon0[:, i])
        lat0i[:, i] = resample_azim(i2_dt, i0_dt, lat0[:, i])
    # subswath 1
    lon1i = np.empty((lon2.shape[0], lon1.shape[1]))
    lat1i = np.empty((lat2.shape[0], lat1.shape[1]))
    for i in range(lon1.shape[1]):
        lon1i[:, i] = resample_azim(i2_dt, i1_dt, lon1[:, i])
        lat1i[:, i] = resample_azim(i2_dt, i1_dt, lat1[:, i])
    # subswath 3
    lon3i = np.empty((lon2.shape[0], lon3.shape[1]))
    lat3i = np.empty((lat2.shape[0], lat3.shape[1]))
    for i in range(lon3.shape[1]):
        lon3i[:, i] = resample_azim(i2_dt, i3_dt, lon3[:, i])
        lat3i[:, i] = resample_azim(i2_dt, i3_dt, lat3[:, i])
    # subswath 4
    lon4i = np.empty((lon2.shape[0], lon4.shape[1]))
    lat4i = np.empty((lat2.shape[0], lat4.shape[1]))
    for i in range(lon4.shape[1]):
        lon4i[:, i] = resample_azim(i2_dt, i4_dt, lon4[:, i])
        lat4i[:, i] = resample_azim(i2_dt, i4_dt, lat4[:, i])

    lon_merged = np.concatenate((lon0i, lon1i, lon2, lon3i, lon4i), axis=1)
    lat_merged = np.concatenate((lat0i, lat1i, lat2, lat3i, lat4i), axis=1)

    va0 = nn[0]["sensor_view_corrected"]
    va0i = np.empty((lon2.shape[0], lon0.shape[1]))
    for i in range(lon0.shape[1]):
        va0i[:, i] = resample_azim(i2_dt, i0_dt, va0[:, i])
    va1 = nn[1]["sensor_view_corrected"]
    va1i = np.empty((lon2.shape[0], lon1.shape[1]))
    for i in range(lon1.shape[1]):
        va1i[:, i] = resample_azim(i2_dt, i1_dt, va1[:, i])
    va2i = nn[2]["sensor_view_corrected"]
    va3 = nn[3]["sensor_view_corrected"]
    va3i = np.empty((lon2.shape[0], lon3.shape[1]))
    for i in range(lon3.shape[1]):
        va3i[:, i] = resample_azim(i2_dt, i3_dt, va3[:, i])
    va4 = nn[4]["sensor_view_corrected"]
    va4i = np.empty((lon2.shape[0], lon4.shape[1]))
    for i in range(lon4.shape[1]):
        va4i[:, i] = resample_azim(i2_dt, i4_dt, va4[:, i])
    va_merged = np.concatenate((va0i, va1i, va2i, va3i, va4i), axis=1)
    # Get index array of sorted view angles (increasing along range)
    indarr = np.argsort(va_merged, axis=1)

    # Create merged Nansat object
    merged = Nansat.from_domain(Domain.from_lonlat(
        np.take_along_axis(lon_merged.astype("float32"), indarr, axis=1),
        np.take_along_axis(lat_merged.astype("float32"), indarr, axis=1),
        add_gcps=False))

    merged.add_band(array=np.take_along_axis(va_merged, indarr, axis=1),
                    parameters={
                        "name": "sensor_view_angle",
                        "long_name": "Sensor View Angle",
                        "standard_name": "sensor_view_angle",
                        "units": "degree",
                        "dataType": 6,
                        "minmax": "15 45",
                        "colormap": "cmocean.cm.gray"})

    ysamplefreq_max = np.round(np.max([
        gg.getinfo(channel=0).gate[0]["YSAMPLEFREQ"],
        gg.getinfo(channel=1).gate[0]["YSAMPLEFREQ"],
        gg.getinfo(channel=2).gate[0]["YSAMPLEFREQ"],
        gg.getinfo(channel=3).gate[0]["YSAMPLEFREQ"],
        gg.getinfo(channel=4).gate[0]["YSAMPLEFREQ"],
    ]))
    bands = {
        "incidence_angle": {
            "minmax": "15 45",
            "colormap": "cmocean.cm.gray",
        },
        "sensor_azimuth": {
            "minmax": "15 45",
            "colormap": "cmocean.cm.gray",
        },
        "sigma0_VV": {
            "colormap": "cmocean.cm.gray",
        },
        "sigma0_HH": {
            "colormap": "cmocean.cm.gray",
        },
        "dc_VV": {
            "minmax": "0 {:d}".format(int(ysamplefreq_max)),
            "colormap": "cmocean.cm.phase",
        },
        "dc_HH": {
            "minmax": "0 {:d}".format(int(ysamplefreq_max)),
            "colormap": "cmocean.cm.phase",
        },
        "dc_std_VV": {
            "minmax": "0 5",
            "colormap": "cmocean.cm.thermal",
        },
        "dc_std_HH": {
            "minmax": "0 5",
            "colormap": "cmocean.cm.thermal",
        },
        "topographic_height": {
            "colormap": "cmocean.cm.topo",
        },
        "valid_land_doppler": {
            "colormap": "cmocean.cm.gray",
        },
        "valid_sea_doppler": {
            "colormap": "cmocean.cm.gray",
        },
        "valid_doppler": {
            "colormap": "cmocean.cm.gray",
        },
        "fe": {
            "minmax": "-200 200",
            "colormap": "cmocean.cm.delta",
        },
        "fgeo": {
            "minmax": "-200 200",
            "colormap": "cmocean.cm.delta",
        },
        "fdg": {
            "minmax": "-60 60",
            "colormap": "cmocean.cm.balance",
        },
        "fww": {
            "minmax": "-60 60",
            "colormap": "cmocean.cm.delta",
        },
        "std_fww": {
            "minmax": "0 10",
            "colormap": "cmocean.cm.thermal",
        },
        "u_range": {
            "minmax": "-1.5 1.5",
            "colormap": "cmocean.cm.delta",
        },
        "std_u_range": {
            "minmax": "0 0.8",
            "colormap": "cmocean.cm.thermal",
        },
        "wind_direction": {
            "minmax": "0 360",
            "colormap": "cmocean.cm.phase",
        },
        "wind_speed": {
            "minmax": "0 20",
            "colormap": "cmocean.cm.speed",
        },
    }

    for band in bands.keys():
        if not nn[0].has_band(band):
            continue
        data0i = np.empty((lon2.shape[0], lon0.shape[1]))
        for i in range(lon0.shape[1]):
            data0i[:, i] = np.interp(i2_dt, i0_dt, nn[0][band][:, i], left=np.nan, right=np.nan)
        data1i = np.empty((lon2.shape[0], lon1.shape[1]))
        for i in range(lon1.shape[1]):
            data1i[:, i] = np.interp(i2_dt, i1_dt, nn[1][band][:, i], left=np.nan, right=np.nan)
        data2i = nn[2][band]
        data3i = np.empty((lon2.shape[0], lon3.shape[1]))
        for i in range(lon3.shape[1]):
            data3i[:, i] = np.interp(i2_dt, i3_dt, nn[3][band][:, i], left=np.nan, right=np.nan)
        data4i = np.empty((lon2.shape[0], lon4.shape[1]))
        for i in range(lon4.shape[1]):
            data4i[:, i] = np.interp(i2_dt, i4_dt, nn[4][band][:, i], left=np.nan, right=np.nan)

        data_merged = np.concatenate((data0i, data1i, data2i, data3i, data4i), axis=1)
        params = nn[0].get_metadata(band_id=band)
        if "valid" in params["name"]:
            params["dataType"] = 3
        else:
            params["dataType"] = 6
        if bands[band].get("minmax", None) is not None:
            params["minmax"] = bands[band]["minmax"]
        params["colormap"] = bands[band]["colormap"]
        merged.add_band(array=np.take_along_axis(data_merged, indarr, axis=1),
                        parameters=params)

    # Add global metadata
    pol = ds.sardopplerextrametadata_set.get().polarization
    merged.set_metadata(key='id', value=ds.entry_id)
    merged.set_metadata(key='naming_authority', value="no.met")
    # TODO: update filename to agreed ESA standard
    esa_fn = "ASA_WSM_1PNPDE20081102_020706_000001162073_00275_34898_8435.N1"
    esa_fn = "XXXXXXXXXX_PDDDDDDDD_TTTTTT_ddddd_PPPPP_ooooo_00000_cccc.N1"
    esa_fn = "ASA_WSDH2P_x20081102_020706_00000_PPPPP_ooooo_00000_cccc.N1"

    orbit_LUT = os.path.join(os.getenv("SAT_AUX_PATH"), "EnvisatMissionOrbits.csv")
    orbits = {}
    with open(orbit_LUT, newline="") as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            orbits[row["PredictedTime"]] = row
    times = [datetime.datetime.strptime(tt, "%d/%m/%Y %H:%M:%S").replace(tzinfo=timezone.utc)
             for tt in list(orbits)]
    delta_t = np.array(times)-ds.time_coverage_start
    orbit_info = orbits[list(orbits)[delta_t[delta_t < datetime.timedelta(0)].argmax()]]

    esa_fn = "ASA_WSD{:s}2PRNMI{:%Y%m%d_%H%M%S}_{:08d}{:d}{:03d}_{:05d}_{:05d}_0000.nc".format(
        pol[0], t0, int(i2_dt[-1]*10**3), int(orbit_info["Phase"]), int(orbit_info["Cycle"]),
        int(orbit_info["RelOrbit"]), int(orbit_info["AbsOrbno"]))
    merged.filename = path_to_merged(ds, esa_fn)
    merged.set_metadata(key='originating_file',
                        value=nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri))
    merged.set_metadata(key='time_coverage_start',
                        value=t0.replace(tzinfo=pytz.utc).isoformat())
    merged.set_metadata(key='time_coverage_end',
                        value=(t0 + datetime.timedelta(seconds=i2_dt[-1])
                               ).replace(tzinfo=pytz.utc).isoformat())

    title = (
        'Calibrated geophysical %s %s wide-swath range Doppler frequency '
        'shift retrievals in %s polarisation, %s') % (
            pti.get_gcmd_platform('envisat')['Short_Name'],
            pti.get_gcmd_instrument('asar')['Short_Name'],
            pol,
            merged.get_metadata('time_coverage_start'))
    merged.set_metadata(key='title', value=title)
    title_no = (
        'Kalibrert geofysisk %s %s Dopplerskift i full bildebredde og '
        '%s polarisering, %s') % (
            pti.get_gcmd_platform('envisat')['Short_Name'],
            pti.get_gcmd_instrument('asar')['Short_Name'],
            pol,
            merged.get_metadata('time_coverage_start'))

    summary = (
        "Calibrated geophysical range Doppler frequency shift "
        "retrievals from an %s %s wide-swath acqusition "
        "obtained on %s. The geophysical Doppler shift "
        "depends on the ocean wave-state and the sea surface "
        "current. In the absence of current, the geophysical "
        "Doppler shift is mostly related to the local wind "
        "speed and direction. The present dataset is in %s "
        "polarization.") % (
            pti.get_gcmd_platform('envisat')['Short_Name'],
            pti.get_gcmd_instrument('asar')['Short_Name'],
            merged.get_metadata('time_coverage_start'),
            ds.sardopplerextrametadata_set.get().polarization)
    merged.set_metadata(key='summary', value=summary)
    summary_no = (
        "Kalibrert geofysisk Dopplerskift fra %s %s målt %s. "
        "Det geofysiske Dopplerskiftet avhenger av "
        "havbølgetilstand og overflatestrøm. Ved fravær av "
        "strøm er det geofysiske Dopplerskiftet stort sett "
        "relatert til den lokale vindhastigheten og dens "
        "retning. Foreliggende datasett er i %s "
        "polarisering.") % (
            pti.get_gcmd_platform('envisat')['Short_Name'],
            pti.get_gcmd_instrument('asar')['Short_Name'],
            merged.get_metadata('time_coverage_start'),
            ds.sardopplerextrametadata_set.get().polarization)

    merged.set_metadata(key="history",
                        value=create_history_message("sar_doppler.utils.create_merged_swaths(ds, ",
                                                     EPSG=EPSG, **kwargs))

    return merged, {"title_no": title_no, "summary_no": summary_no}
