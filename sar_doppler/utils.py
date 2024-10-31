"""
Utility functions for processing Doppler from multiple SAR acquisitions
"""
import os
import csv
import pytz
import logging
import netCDF4
import pathlib
import datetime

import numpy as np
import xarray as xr

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from py_mmd_tools import nc_to_mmd

import pythesint as pti

from nansat.domain import Domain
from nansat.nansat import Nansat

from django.conf import settings

from django.db import connection
from django.db.utils import OperationalError

from django.utils import timezone

from geospaas.catalog.models import Dataset
from geospaas.catalog.models import DatasetURI

from sardoppler.gsar import gsar
from sardoppler.sardoppler import Doppler
from sardoppler.sardoppler import surface_radial_doppler_sea_water_velocity

from geospaas.utils.utils import nansat_filename
from geospaas.utils.utils import product_path


offset_corr_types = {
    "land": Doppler.LAND_OFFSET_CORRECTION,
    "cdop": Doppler.CDOP_OFFSET_CORRECTION,
    "aligned_subswath_edges": Doppler.ALIGNED_SUBSWATHS,
    "none": Doppler.NO_OFFSET_CORRECTION,
}
inverse_offset_corr_types = {v: k for k, v in offset_corr_types.items()}


def find_wind(ds):
    """Find ERA5 reanalysis wind collocation with the given dataset.
    """
    db_locked = True
    while db_locked:
        try:
            wind_fn = nansat_filename(
                Dataset.objects.get(
                    source__platform__short_name='ERA15DAS',
                    time_coverage_start__lte=ds.time_coverage_end,
                    time_coverage_end__gte=ds.time_coverage_start
                ).dataseturi_set.get().uri
            )
        except OperationalError:
            db_locked = True
        except Exception as e:
            logging.error("%s - in search for ERA15DAS data (%s, %s, %s) " % (
                str(e),
                nansat_filename(ds.dataseturi_set.get(uri__endswith=".gsar").uri),
                ds.time_coverage_start,
                ds.time_coverage_end
            ))
            return None
        else:
            db_locked = False
    connection.close()

    return wind_fn


def plot_map(n, band="geophysical_doppler", vmin=-60, vmax=60, title=None, cmap=None):
    """ Plot a map of the given band.
    """
    import cmocean
    if cmap is None:
        cmap = eval(n.get_metadata(band_id="geophysical_doppler", key="colormap"))

    lon, lat = n.get_geolocation_grids()

    # FIG 1
    if lon.max() - lon.min() > 100:
        ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        lon = np.mod(lon, 360) - 180
    else:
        ax1 = plt.subplot(projection=ccrs.PlateCarree())

    cb = True

    da = xr.DataArray(n[band], dims=["y", "x"],
                      coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})
    da.plot.pcolormesh("lon", "lat", ax=ax1, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=cb)

    ax1.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
    ax1.gridlines(draw_labels=True)
    # if title is None:
    #     plt.title("Wind on %s" % n.time_coverage_start.strftime("%Y-%m-%d"))

    plt.show()


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
            if isinstance(arg, str):
                history_message += "\'%s\', " % arg
            else:
                history_message += "%s, " % str(arg)
    if bool(kwargs):
        for key in kwargs.keys():
            if kwargs[key]:
                if isinstance(kwargs[key], str):
                    history_message += "%s=\'%s\', " % (key, kwargs[key])
                else:
                    history_message += "%s=%s, " % (key, str(kwargs[key]))
    return history_message[:-2] + ")"


def module_name():
    """ Get module name
    """
    return __name__.split(".")[0]


def path_to_merged(ds, fn):
    """Get the path to a merged file with basename fn.
    """
    pp = product_path(module_name() + ".merged", "", date=ds.time_coverage_start)
    connection.close()
    return os.path.join(pp, os.path.basename(fn))


def path_to_nc_products(ds):
    """ Get the (product) path to netCDF-CF files."""
    pp = product_path(module_name(),
                      nansat_filename(get_dataseturi_uri_endswith(ds, ".gsar").uri),
                      date=ds.time_coverage_start)
    connection.close()
    return pp


def path_to_nc_file(ds, fn):
    """ Get the path to a netcdf product with filename fn."""
    return os.path.join(path_to_nc_products(ds), os.path.basename(fn))


def nc_name(ds, ii):
    """ Get the full path filename of exported netcdf of subswath ii."""
    fn = path_to_nc_file(ds, os.path.basename(nansat_filename(
        get_dataseturi_uri_endswith(ds, ".gsar").uri)).split(".")[0] + "subswath%s.nc" % ii)
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


def create_mmd_file(ds, uri, check_only=False):
    """Create MMD files for the provided dataset nc uri."""
    base_url = "https://thredds.met.no/thredds/dodsC/remotesensingenvisat/asar-doppler"
    dataset_citation = {
        "author": "Morten W. Hansen, Jeong-Won Park, Geir Engen, Harald Johnsen",
        "publication_date": "2023-10-05",
        "title": "Calibrated geophysical ENVISAT ASAR wide-swath range Doppler frequency shift",
        "publisher":
            "European Space Agency (ESA), Norwegian Meteorological Institute (MET Norway)",
            "url": "https://data.met.no/dataset/{:s}".format(ds.entry_id),
        "doi": "https://doi.org/10.57780/esa-56fb232"
    }

    outfile = os.path.join(
        product_path(module_name() + ".mmd", "", date=ds.time_coverage_start),
        pathlib.Path(pathlib.Path(nansat_filename(uri.uri)).stem).with_suffix(".xml")
    )

    odap_url = base_url + uri.uri.split("sar_doppler/merged")[-1]
    wms_base_url = "https://fastapi.s-enda-staging.k8s.met.no/api/get_quicklook"

    path_parts = nansat_filename(uri.uri).split("/")
    year = path_parts[-4]
    month = path_parts[-3]
    day = path_parts[-2]
    wms_url = os.path.join(wms_base_url, year, month, day, path_parts[-1])
    layers = ["geophysical_doppler",
              "electronic_mispointing",
              "geometric_doppler",
              "wind_waves_doppler",
              "wind_direction",
              "wind_speed",
              "ground_range_current",
              "std_ground_range_current",
              "topographic_height",
              "valid_doppler"]

    orbit_info = get_orbit_info(ds.time_coverage_start)
    dop = netCDF4.Dataset(nansat_filename(uri.uri))

    platform = {
        "short_name": "Envisat",
        "long_name": "Environmental Satellite",
        "resource": "https://space.oscar.wmo.int/satelliteprogrammes/view/envisat",
        "orbit_relative": orbit_info["RelOrbit"],
        "orbit_absolute": orbit_info["AbsOrbno"],
        "orbit_direction": dop.orbit_direction,
        "instrument": {
            "short_name": "ASAR",
            "long_name": "Advanced Synthetic Aperture Radar",
            "resource": "https://space.oscar.wmo.int/instruments/view/asar",
            "polarisation": ds.sardopplerextrametadata_set.get().polarization,
        }
    }

    logging.info("Creating MMD file: %s" % outfile)
    md = nc_to_mmd.Nc_to_mmd(nansat_filename(uri.uri), opendap_url=odap_url,
                             output_file=outfile, check_only=check_only)
    req_ok, msg = md.to_mmd(checksum_calculation=True,
                            parent="no.met:e19b9c36-a9dc-4e13-8827-c998b9045b54",
                            add_wms_data_access=True,
                            wms_link=wms_url,
                            wms_layer_names=layers,
                            overrides={
                                "dataset_citation": dataset_citation,
                                "platform": platform,
                            })
    # Add MMD to dataseturis
    mmd_uri = "file://localhost" + outfile
    locked = True
    while locked:
        try:
            new_uri, created = DatasetURI.objects.get_or_create(uri=mmd_uri, dataset=ds)
        except OperationalError:
            locked = True
        else:
            locked = False
    connection.close()
    return new_uri, created


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
    nc_uris = ds.dataseturi_set.filter(uri__contains=".nc")
    if nc_uris:
        nc_uris = nc_uris.filter(uri__contains="subswath")
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
            nansat_filename(uri.uri)).split(".")[0])
    return ds, proc


def get_orbit_info(time_coverage_start):
    orbit_LUT = os.path.join(os.getenv("SAT_AUX_PATH"), "EnvisatMissionOrbits.csv")
    orbits = {}
    with open(orbit_LUT, newline="") as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            orbits[row["PredictedTime"]] = row
    times = [datetime.datetime.strptime(tt, "%d/%m/%Y %H:%M:%S").replace(tzinfo=timezone.utc)
             for tt in list(orbits)]
    delta_t = np.array(times) - time_coverage_start
    return orbits[list(orbits)[delta_t[delta_t < datetime.timedelta(0)].argmax()]]


def get_dataseturi_uri_endswith(ds, ending):
    """Small function to omit locked db.
    """
    locked = True
    while locked:
        try:
            uri = ds.dataseturi_set.get(uri__endswith=ending)
        except OperationalError:
            locked = True
        else:
            locked = False
    connection.close()
    return uri


def create_merged_swaths(ds, EPSG=4326, **kwargs):
    """Merge swaths, add dataseturi, and return Nansat object.

    EPSG options:
        - 4326: WGS 84 / longlat
        - 3995: WGS 84 / Arctic Polar Stereographic
    """
    gsar_uri = get_dataseturi_uri_endswith(ds, ".gsar").uri
    logging.info("Merging subswaths of {:s}.".format(gsar_uri))
    nn = {}
    nn[0] = Doppler(nansat_filename(get_dataseturi_uri_endswith(ds, "swath%d.nc" % 0).uri))
    nn[1] = Doppler(nansat_filename(get_dataseturi_uri_endswith(ds, "swath%d.nc" % 1).uri))
    nn[2] = Doppler(nansat_filename(get_dataseturi_uri_endswith(ds, "swath%d.nc" % 2).uri))
    nn[3] = Doppler(nansat_filename(get_dataseturi_uri_endswith(ds, "swath%d.nc" % 3).uri))
    nn[4] = Doppler(nansat_filename(get_dataseturi_uri_endswith(ds, "swath%d.nc" % 4).uri))

    gg = gsar(nansat_filename(gsar_uri))

    connection.close()

    locked = True
    while locked:
        try:
            pol = ds.sardopplerextrametadata_set.get().polarization
        except OperationalError:
            locked = True
        else:
            locked = False
    connection.close()

    # Azimuth times as datetime.datetime
    i0_ytimes = nn[0].get_azimuth_time()
    i1_ytimes = nn[1].get_azimuth_time()
    i2_ytimes = nn[2].get_azimuth_time()
    i3_ytimes = nn[3].get_azimuth_time()
    i4_ytimes = nn[4].get_azimuth_time()

    # Earliest measurement as datetime.datetime
    t0 = np.min(np.array([i0_ytimes[0], i1_ytimes[0], i2_ytimes[0], i3_ytimes[0], i4_ytimes[0]]))

    helper = np.vectorize(lambda x: x.total_seconds())
    i0_dt = helper(i0_ytimes - t0)
    i1_dt = helper(i1_ytimes - t0)
    i2_dt = helper(i2_ytimes - t0)
    i3_dt = helper(i3_ytimes - t0)
    i4_dt = helper(i4_ytimes - t0)

    lon0, lat0 = nn[0].get_geolocation_grids()
    lon1, lat1 = nn[1].get_geolocation_grids()
    lon2, lat2 = nn[2].get_geolocation_grids()
    lon3, lat3 = nn[3].get_geolocation_grids()
    lon4, lat4 = nn[4].get_geolocation_grids()

    test_dateline = np.array([
        lon0.max() - lon0.min() > 300,
        lon1.max() - lon1.min() > 300,
        lon2.max() - lon2.min() > 300,
        lon3.max() - lon3.min() > 300,
        lon4.max() - lon4.min() > 300])
    """This should not be necessary, since the plotting functions
    need to handle it, but it needs to be done for the interpolation
    further down. In some lines the first value will be near 180
    whereas the next will be near -180, and interpolation will
    result in something around 0 unless a modulo is done.
    """
    if np.any(test_dateline):
        lon0 = np.mod(lon0, 360)
        lon1 = np.mod(lon1, 360)
        lon2 = np.mod(lon2, 360)
        lon3 = np.mod(lon3, 360)
        lon4 = np.mod(lon4, 360)

    """Some times the subswaths 1, 2, 4 and 5 are shorter in azimuth
    than subswath 3. In this case, we need to extrapolate the lon,
    lat and view_angle grids. This is challenging, so it easier to
    throw data outside the minimum maximum azimuth time, and the
    maximum minimum time of the swaths.

    We may want to extrapolate later in order to not throw away data..
    """
    throw = i2_dt < i0_dt[0]
    throw[i2_dt > i0_dt[-1]] = True
    throw[i2_dt < i1_dt[0]] = True
    throw[i2_dt > i1_dt[-1]] = True
    throw[i2_dt < i3_dt[0]] = True
    throw[i2_dt > i3_dt[-1]] = True
    throw[i2_dt < i4_dt[0]] = True
    throw[i2_dt > i4_dt[-1]] = True

    if len(throw[throw]) > 10 or len(throw[throw]) > len(throw[throw == False]):
        if len(throw[throw]) > 10:
            detail = "The azimuth length difference between " \
                     "subswaths is greater than 10 lines."
        else:
            detail = "The azimuth length of the shortest subswath " \
                     "is less than half the length of the third one."
        logging.warning(f"Possible erroneous subswath in {ds.entry_id}. {detail}")

    i2_dt = np.delete(i2_dt, throw)

    def valid_resampled(xnew, x, y):
        valid = np.ones(xnew.shape)
        valid[xnew < x[0]] = 0
        valid[xnew > x[-1]] = 0
        return valid

    def resample_azim(xnew, x, y):
        yy = np.interp(xnew, x, y, left=np.nan, right=np.nan)
        xs = xnew[~np.isnan(yy)]
        ys = yy[~np.isnan(yy)]
        z = CubicSpline(xs, ys, bc_type="natural")
        ynew = z(xnew, nu=0)
        return ynew

    # Interpolate lon/lat
    # subswath 0
    lon0i = np.empty((i2_dt.size, lon0.shape[1]))
    lat0i = np.empty((i2_dt.size, lat0.shape[1]))
    valid0i = np.empty((i2_dt.size, lat0.shape[1]))
    for i in range(lon0.shape[1]):
        lon0i[:, i] = resample_azim(i2_dt, i0_dt, lon0[:, i])
        lat0i[:, i] = resample_azim(i2_dt, i0_dt, lat0[:, i])
        valid0i[:, i] = valid_resampled(i2_dt, i0_dt, lat0[:, i])
    # subswath 1
    lon1i = np.empty((i2_dt.size, lon1.shape[1]))
    lat1i = np.empty((i2_dt.size, lat1.shape[1]))
    valid1i = np.empty((i2_dt.size, lat1.shape[1]))
    for i in range(lon1.shape[1]):
        lon1i[:, i] = resample_azim(i2_dt, i1_dt, lon1[:, i])
        lat1i[:, i] = resample_azim(i2_dt, i1_dt, lat1[:, i])
        valid1i[:, i] = valid_resampled(i2_dt, i1_dt, lat1[:, i])
    # subswath 3
    lon3i = np.empty((i2_dt.size, lon3.shape[1]))
    lat3i = np.empty((i2_dt.size, lat3.shape[1]))
    valid3i = np.empty((i2_dt.size, lat3.shape[1]))
    for i in range(lon3.shape[1]):
        lon3i[:, i] = resample_azim(i2_dt, i3_dt, lon3[:, i])
        lat3i[:, i] = resample_azim(i2_dt, i3_dt, lat3[:, i])
        valid3i[:, 1] = valid_resampled(i2_dt, i3_dt, lat3[:, i])
    # subswath 4
    lon4i = np.empty((i2_dt.size, lon4.shape[1]))
    lat4i = np.empty((i2_dt.size, lat4.shape[1]))
    valid4i = np.empty((i2_dt.size, lat4.shape[1]))
    for i in range(lon4.shape[1]):
        lon4i[:, i] = resample_azim(i2_dt, i4_dt, lon4[:, i])
        lat4i[:, i] = resample_azim(i2_dt, i4_dt, lat4[:, i])
        valid4i[:, i] = valid_resampled(i2_dt, i4_dt, lat4[:, i])

    # Update lon2 and lat2
    lon2i = lon2[throw == False, :]
    lat2i = lat2[throw == False, :]
    lon_merged = np.concatenate((lon0i, lon1i, lon2i, lon3i, lon4i), axis=1)
    # Change back to longitude range between -180 and 180 degrees
    lon_merged = np.mod(lon_merged - 180., 360.) - 180.
    lat_merged = np.concatenate((lat0i, lat1i, lat2i, lat3i, lat4i), axis=1)

    va0 = nn[0]["sensor_view_corrected"]
    va0i = np.empty((i2_dt.size, lon0.shape[1]))
    nrcs0 = nn[0].corrected_sigma0()
    nrcs0i = np.empty((i2_dt.size, lon0.shape[1]))
    for i in range(lon0.shape[1]):
        va0i[:, i] = resample_azim(i2_dt, i0_dt, va0[:, i])
        nrcs0i[:, i] = resample_azim(i2_dt, i0_dt, nrcs0[:, i])

    va1 = nn[1]["sensor_view_corrected"]
    va1i = np.empty((i2_dt.size, lon1.shape[1]))
    nrcs1 = nn[1].corrected_sigma0()
    nrcs1i = np.empty((i2_dt.size, lon1.shape[1]))
    for i in range(lon1.shape[1]):
        va1i[:, i] = resample_azim(i2_dt, i1_dt, va1[:, i])
        nrcs1i[:, i] = resample_azim(i2_dt, i1_dt, nrcs1[:, i])

    va2i = nn[2]["sensor_view_corrected"][throw == False, :]
    nrcs2i = nn[2].corrected_sigma0()[throw == False, :]

    va3 = nn[3]["sensor_view_corrected"]
    va3i = np.empty((i2_dt.size, lon3.shape[1]))
    nrcs3 = nn[3].corrected_sigma0()
    nrcs3i = np.empty((i2_dt.size, lon3.shape[1]))
    for i in range(lon3.shape[1]):
        va3i[:, i] = resample_azim(i2_dt, i3_dt, va3[:, i])
        nrcs3i[:, i] = resample_azim(i2_dt, i3_dt, nrcs3[:, i])

    va4 = nn[4]["sensor_view_corrected"]
    va4i = np.empty((i2_dt.size, lon4.shape[1]))
    nrcs4 = nn[4].corrected_sigma0()
    nrcs4i = np.empty((i2_dt.size, lon4.shape[1]))
    for i in range(lon4.shape[1]):
        va4i[:, i] = resample_azim(i2_dt, i4_dt, va4[:, i])
        nrcs4i[:, i] = resample_azim(i2_dt, i4_dt, nrcs4[:, i])

    va_merged = np.concatenate((va0i, va1i, va2i, va3i, va4i), axis=1)
    # Get index array of sorted view angles (increasing along range)
    indarr = np.argsort(va_merged, axis=1)

    nrcs_merged = np.take_along_axis(np.concatenate((nrcs0i, nrcs1i, nrcs2i, nrcs3i, nrcs4i),
                                                    axis=1),
                                     indarr, axis=1)

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
                        "grid_mapping": "crs",
                        "minmax": "15 45",
                        "colormap": "cmocean.cm.gray"})

    merged.add_band(array=nrcs_merged,
                    parameters={
                        "name": "sigma0",
                        "short_name": "RCS",
                        "long_name": "Radar Cross Section (not calibrated)",
                        "standard_name": "surface_backwards_scattering_coefficient_of_radar_wave",
                        "units": "m/m",
                        "polarization": pol,
                        "colormap": "cmocean.cm.gray",
                        "grid_mapping": "crs",
                        "dataType": 6})

    # Create array indicating which subswath the pixels belong to
    ss1 = np.ones(va0i.shape)
    ss2 = np.ones(va1i.shape)*2
    ss3 = np.ones(va2i.shape)*3
    ss4 = np.ones(va3i.shape)*4
    ss5 = np.ones(va4i.shape)*5
    subswaths = np.take_along_axis(np.concatenate((ss1, ss2, ss3, ss4, ss5), axis=1),
                                   indarr, axis=1)
    merged.add_band(array=subswaths,
                    parameters={
                        "name": "subswaths",
                        "long_name": "per pixel subswath number",
                        "dataType": 3,
                        "grid_mapping": "crs",
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
        "dc": {
            "minmax": "0 {:d}".format(int(ysamplefreq_max)),
            "colormap": "cmocean.cm.phase",
        },
        "dc_std": {
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
        "electronic_mispointing": {
            "minmax": "-200 200",
            "colormap": "cmocean.cm.balance",
        },
        "geometric_doppler": {
            "minmax": "-200 200",
            "colormap": "cmocean.cm.balance",
        },
        "geophysical_doppler": {
            "minmax": "-60 60",
            "colormap": "cmocean.cm.balance",
        },
        "wind_waves_doppler": {
            "minmax": "-60 60",
            "colormap": "cmocean.cm.balance",
        },
        "std_wind_waves_doppler": {
            "minmax": "0 10",
            "colormap": "cmocean.cm.thermal",
        },
        "ground_range_current": {
            "minmax": "-1.5 1.5",
            "colormap": "cmocean.cm.balance",
        },
        "std_ground_range_current": {
            "minmax": "0 0.8",
            "colormap": "cmocean.cm.thermal",
        },
        "wind_direction": {
            "minmax": "0 360",
            "colormap": "cmocean.cm.phase",
        },
        "wind_speed": {
            "minmax": "0 20",
            "colormap": "cmocean.cm.haline",
        },
    }

    for band in bands.keys():
        if band in ["ground_range_current", "std_ground_range_current"]:
            continue
        if not nn[0].has_band(band):
            continue
        data0i = np.empty((i2_dt.size, lon0.shape[1]))
        for i in range(lon0.shape[1]):
            data0i[:, i] = np.interp(i2_dt, i0_dt, nn[0][band][:, i], left=np.nan, right=np.nan)
        data1i = np.empty((i2_dt.size, lon1.shape[1]))
        for i in range(lon1.shape[1]):
            data1i[:, i] = np.interp(i2_dt, i1_dt, nn[1][band][:, i], left=np.nan, right=np.nan)
        data2i = nn[2][band][throw == False, :]
        data3i = np.empty((i2_dt.size, lon3.shape[1]))
        for i in range(lon3.shape[1]):
            # resample_azim not done since nan here is ok. Lon/lat
            # have to be real on the other hand.
            # data3i[:, i] = resample_azim(i2_dt, i3_dt, n[3][band][:, i])
            data3i[:, i] = np.interp(i2_dt, i3_dt, nn[3][band][:, i], left=np.nan, right=np.nan)
        data4i = np.empty((i2_dt.size, lon4.shape[1]))
        for i in range(lon4.shape[1]):
            data4i[:, i] = np.interp(i2_dt, i4_dt, nn[4][band][:, i], left=np.nan, right=np.nan)

        data_merged = np.take_along_axis(np.concatenate((data0i, data1i, data2i, data3i, data4i),
                                                        axis=1),
                                         indarr, axis=1)
        params = nn[0].get_metadata(band_id=band)
        params.pop("wkv", "")
        if band == "geophysical_doppler":
            fdg = data_merged
            continue
        if "valid" in params["name"]:
            params["dataType"] = 3
        else:
            params["dataType"] = 6
        if bands[band].get("minmax", None) is not None:
            params["minmax"] = bands[band]["minmax"]
        params["colormap"] = bands[band]["colormap"]
        params["grid_mapping"] = "crs"
        merged.add_band(array=data_merged, parameters=params)

    # Add global metadata
    merged.set_metadata(key="id", value=ds.entry_id)
    merged.set_metadata(key="naming_authority", value="no.met")

    orbit_info = get_orbit_info(ds.time_coverage_start)

    # filename according to agreed ESA standard
    esa_fn = "ASA_WSD{:s}2PRNMI{:%Y%m%d_%H%M%S}_{:08d}{:d}{:03d}_{:05d}_{:05d}_0000.nc".format(
        pol[0], t0, int(i2_dt[-1]*10**3), int(orbit_info["Phase"]), int(orbit_info["Cycle"]),
        int(orbit_info["RelOrbit"]), int(orbit_info["AbsOrbno"]))
    merged.filename = path_to_merged(ds, esa_fn)
    merged.set_metadata(key="originating_file",
                        value=nansat_filename(gsar_uri))
    merged.set_metadata(key="time_coverage_start",
                        value=t0.replace(tzinfo=pytz.utc).isoformat())
    merged.set_metadata(key="time_coverage_end",
                        value=(t0 + datetime.timedelta(seconds=i2_dt[-1])
                               ).replace(tzinfo=pytz.utc).isoformat())

    title = (
        "Calibrated geophysical %s %s wide-swath range Doppler frequency "
        "shift retrievals in %s polarisation, %s") % (
            pti.get_gcmd_platform("envisat")["Short_Name"],
            pti.get_gcmd_instrument("asar")["Short_Name"],
            pol,
            merged.get_metadata("time_coverage_start"))
    merged.set_metadata(key="title", value=title)
    title_no = (
        "Kalibrert geofysisk %s %s Dopplerskift i full bildebredde og "
        "%s polarisering, %s") % (
            pti.get_gcmd_platform("envisat")["Short_Name"],
            pti.get_gcmd_instrument("asar")["Short_Name"],
            pol,
            merged.get_metadata("time_coverage_start"))

    summary = (
        "Calibrated geophysical range Doppler frequency shift "
        "retrievals from an %s %s wide-swath acqusition "
        "obtained on %s. The geophysical Doppler shift "
        "depends on the ocean wave-state and the sea surface "
        "current. In the absence of current, the geophysical "
        "Doppler shift is mostly related to the local wind "
        "speed and direction. The present dataset is in %s "
        "polarization.") % (
            pti.get_gcmd_platform("envisat")["Short_Name"],
            pti.get_gcmd_instrument("asar")["Short_Name"],
            merged.get_metadata("time_coverage_start"),
            pol)
    merged.set_metadata(key="summary", value=summary)
    summary_no = (
        "Kalibrert geofysisk Dopplerskift fra %s %s målt %s. "
        "Det geofysiske Dopplerskiftet avhenger av "
        "havbølgetilstand og overflatestrøm. Ved fravær av "
        "strøm er det geofysiske Dopplerskiftet stort sett "
        "relatert til den lokale vindhastigheten og dens "
        "retning. Foreliggende datasett er i %s "
        "polarisering.") % (
            pti.get_gcmd_platform("envisat")["Short_Name"],
            pti.get_gcmd_instrument("asar")["Short_Name"],
            merged.get_metadata("time_coverage_start"),
            pol)

    gg = gsar(nansat_filename(gsar_uri))
    lat = gg.getdata(channel=0)["LATITUDE"]
    assert i0_ytimes[-1] > i0_ytimes[0]
    merged.set_metadata(key="orbit_direction",
                        value="descending" if np.median(np.gradient(lat)) < 0 else "ascending")

    merged.set_metadata(key="history",
                        value=create_history_message("sar_doppler.utils.create_merged_swaths(ds, ",
                                                     EPSG=EPSG, **kwargs))

    # Do offset correction of fdg, and add band
    initial_offset_values = "%s, %s, %s, %s, %s" % (
        nn[0].get_metadata(band_id="geophysical_doppler", key="initial_offset_value"),
        nn[1].get_metadata(band_id="geophysical_doppler", key="initial_offset_value"),
        nn[2].get_metadata(band_id="geophysical_doppler", key="initial_offset_value"),
        nn[3].get_metadata(band_id="geophysical_doppler", key="initial_offset_value"),
        nn[4].get_metadata(band_id="geophysical_doppler", key="initial_offset_value"))
    initial_offset_correction_types = "%s, %s, %s, %s, %s" % (
        nn[0].get_metadata(band_id="geophysical_doppler", key="initial_offset_correction_type"),
        nn[1].get_metadata(band_id="geophysical_doppler", key="initial_offset_correction_type"),
        nn[2].get_metadata(band_id="geophysical_doppler", key="initial_offset_correction_type"),
        nn[3].get_metadata(band_id="geophysical_doppler", key="initial_offset_correction_type"),
        nn[4].get_metadata(band_id="geophysical_doppler", key="initial_offset_correction_type"))
    secondary_offset_values = "%s, %s, %s, %s, %s" % (
        nn[0].get_metadata(band_id="geophysical_doppler", key="secondary_offset_value"),
        nn[1].get_metadata(band_id="geophysical_doppler", key="secondary_offset_value"),
        nn[2].get_metadata(band_id="geophysical_doppler", key="secondary_offset_value"),
        nn[3].get_metadata(band_id="geophysical_doppler", key="secondary_offset_value"),
        nn[4].get_metadata(band_id="geophysical_doppler", key="secondary_offset_value"))
    secondary_offset_correction_types = "%s, %s, %s, %s, %s" % (
        nn[0].get_metadata(band_id="geophysical_doppler", key="secondary_offset_correction_type"),
        nn[1].get_metadata(band_id="geophysical_doppler", key="secondary_offset_correction_type"),
        nn[2].get_metadata(band_id="geophysical_doppler", key="secondary_offset_correction_type"),
        nn[3].get_metadata(band_id="geophysical_doppler", key="secondary_offset_correction_type"),
        nn[4].get_metadata(band_id="geophysical_doppler", key="secondary_offset_correction_type"))
    """The median over land may be slightly biased because some
    subswaths contain more pixels than others, so avoiding a last
    correction:
    """
    # if (merged["valid_land_doppler"] == 1).any():
    #     # Based on land data
    #     offset = np.median(fdg[merged["valid_land_doppler"] == 1])
    #     offset_correction = "land"
    # else:
    land_corrected = [offset_corr_types[corr.strip()] == Doppler.LAND_OFFSET_CORRECTION
                      for corr in initial_offset_correction_types.split(",")]
    tertiary_offset = 0
    tertiary_offset_corr_type = Doppler.NO_OFFSET_CORRECTION
    if not any(land_corrected):
        # Based on CDOP corrected ocean (assuming 0 current)
        no_wind_doppler = fdg[merged["valid_sea_doppler"] == 1] - \
            merged["wind_waves_doppler"][merged["valid_sea_doppler"] == 1]
        tertiary_offset = np.median(no_wind_doppler)
        tertiary_offset_corr_type = Doppler.CDOP_OFFSET_CORRECTION
        fdg -= tertiary_offset
    params = nn[0].get_metadata(band_id="geophysical_doppler")
    params.pop("offset_value", "")
    params.pop("initial_offset_value", "")
    params.pop("initial_offset_correction_type", "")
    params.pop("secondary_offset_value", "")
    params.pop("secondary_offset_correction_type", "")
    params.pop("comment", "")
    params.pop("wkv", "")
    params["dataType"] = 6
    params["minmax"] = bands["geophysical_doppler"]["minmax"]
    params["colormap"] = bands["geophysical_doppler"]["colormap"]
    params["initial_offset_values"] = initial_offset_values
    params["initial_offset_correction_types"] = initial_offset_correction_types
    params["secondary_offset_values"] = secondary_offset_values
    params["secondary_offset_correction_types"] = secondary_offset_correction_types
    params["tertiary_offset_value"] = str(tertiary_offset)
    params["tertiary_offset_correction_type"] = inverse_offset_corr_types[
        tertiary_offset_corr_type]

    merged.add_band(array=fdg, parameters=params)

    # Calculate range current velocity component
    current = surface_radial_doppler_sea_water_velocity(merged)
    merged.add_band(
        array=current[0],
        parameters={"name": "ground_range_current",
                    "long_name": "Sea surface current velocity in ground range direction",
                    "units": "m s-1",
                    "minmax": bands["ground_range_current"]["minmax"],
                    "colormap": bands["ground_range_current"]["colormap"]})
    merged.add_band(
        array=current[1],
        parameters={"name": "std_ground_range_current",
                    "long_name": ("Standard deviation of sea surface current velocity in ground "
                                  "range direction"),
                    "units": "m s-1",
                    "minmax": bands["ground_range_current"]["minmax"],
                    "colormap": bands["ground_range_current"]["colormap"]})

    return merged, {"title_no": title_no, "summary_no": summary_no}
