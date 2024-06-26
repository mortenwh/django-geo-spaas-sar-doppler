import logging
import netCDF4
import os
import subprocess
import uuid

import numpy as np

from datetime import datetime
from math import sin, pi, cos, acos, copysign

from osgeo import ogr
from osgeo import osr
from osgeo import gdal
from osgeo.ogr import Geometry

from django.db import connection
from django.db.utils import OperationalError
from django.utils import timezone
from django.contrib.gis.geos import WKTReader

# Plotting
import matplotlib.pyplot as plt
import cmocean
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# Nansat/geospaas
import pythesint as pti

from geospaas.utils.utils import nansat_filename
from geospaas.utils.utils import media_path
from geospaas.catalog.models import GeographicLocation
from geospaas.catalog.models import Dataset
from geospaas.catalog.models import DatasetURI
from geospaas.nansat_ingestor.managers import DatasetManager as DM

from nansat.nansat import Nansat
from nansat.nsr import NSR
from nansat.domain import Domain

from sardoppler.sardoppler import Doppler
from sardoppler.utils import ASAR_WAVELENGTH

import sar_doppler
from sar_doppler.utils import nansumwrapper
from sar_doppler.utils import create_history_message
from sar_doppler.utils import module_name
from sar_doppler.utils import nc_name
from sar_doppler.utils import lut_results_path
from sar_doppler.utils import path_to_nc_file
from sar_doppler.utils import create_mmd_file
from sar_doppler.utils import create_merged_swaths


# Turn off the error messages completely
gdal.PushErrorHandler('CPLQuietErrorHandler')


def LL2XY(EPSG, lon, lat):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(EPSG)
    coordTransform = osr.CoordinateTransformation(inSpatialRef,
                            outSpatialRef)
    point.Transform(coordTransform)
    return point.GetX(), point.GetY()


def set_fill_value(nobj, bdict, fill_value=9999.):
    """ this does not seem to have any effect..
    But it may be ok anyway: 
        In [14]: netCDF4.default_fillvals
        Out[14]: 
        {'S1': '\x00',
         'i1': -127,
         'u1': 255,
         'i2': -32767,
         'u2': 65535,
         'i4': -2147483647,
         'u4': 4294967295,
         'i8': -9223372036854775806,
         'u8': 18446744073709551614,
         'f4': 9.969209968386869e+36,
         'f8': 9.969209968386869e+36}

    """
    band_num = nobj.get_band_number(bdict)
    rb = nobj.vrt.dataset.GetRasterBand(band_num)
    rb.SetNoDataValue(fill_value)
    rb = None


class DatasetManager(DM):

    N_SUBSWATHS = 5

    def get_or_create(self, uri, *args, **kwargs):
        """ Ingest gsar file to geo-spaas db
        """

        ds, created = super(DatasetManager, self).get_or_create(uri, *args, **kwargs)
        connection.close()

        if not created:
            return ds, False

        fn = nansat_filename(uri)
        n = Nansat(fn, subswath=0)

        # set Dataset entry_title
        ds.entry_title = n.get_metadata('title')
        ds.save()
        connection.close()

        from sar_doppler.models import SARDopplerExtraMetadata
        # Store the polarization and associate the dataset
        extra, _ = SARDopplerExtraMetadata.objects.get_or_create(dataset=ds,
            polarization=n.get_metadata('polarization'))
        if created and not _:
            raise ValueError('Created new dataset but could not '
                             'create instance of ExtraMetadata')
        if _:
            ds.sardopplerextrametadata_set.add(extra)
        connection.close()

        gg = WKTReader().read(n.get_border_wkt())

        #lon, lat = n.get_border()
        #ind_near_range = 0
        #ind_far_range = int(lon.size/4)
        #import pyproj
        #geod = pyproj.Geod(ellps='WGS84')
        #angle1,angle2,img_width = geod.inv(lon[ind_near_range], lat[ind_near_range], 
        #                                    lon[ind_far_range], lat[ind_far_range])

        # If the area of the dataset geometry is larger than the area
        # of the subswath border, it means that the dataset has
        # already been created (the area should be the total area of
        # all subswaths)
        if np.floor(ds.geographic_location.geometry.area) > 2*np.round(gg.area):
            return ds, False

        swath_data = {}
        lon = {}
        lat = {}
        astep = {}
        rstep = {}
        az_left_lon = {}
        ra_upper_lon = {}
        az_right_lon = {}
        ra_lower_lon = {}
        az_left_lat = {}
        ra_upper_lat = {}
        az_right_lat = {}
        ra_lower_lat = {}
        num_border_points = 10
        border = 'POLYGON(('

        for i in range(self.N_SUBSWATHS):
            # Read subswaths 
            swath_data[i] = Nansat(fn, subswath=i)

            lon[i], lat[i] = swath_data[i].get_geolocation_grids()

            astep[i] = int(max(1, (lon[i].shape[0] / 2 * 2 - 1) / num_border_points))
            rstep[i] = int(max(1, (lon[i].shape[1] / 2 * 2 - 1) / num_border_points))

            az_left_lon[i] = lon[i][0:-1:astep[i], 0]
            az_left_lat[i] = lat[i][0:-1:astep[i], 0]

            az_right_lon[i] = lon[i][0:-1:astep[i], -1]
            az_right_lat[i] = lat[i][0:-1:astep[i], -1]

            ra_upper_lon[i] = lon[i][-1, 0:-1:rstep[i]]
            ra_upper_lat[i] = lat[i][-1, 0:-1:rstep[i]]

            ra_lower_lon[i] = lon[i][0, 0:-1:rstep[i]]
            ra_lower_lat[i] = lat[i][0, 0:-1:rstep[i]]

        lons = np.concatenate((az_left_lon[0],  ra_upper_lon[0],
                               ra_upper_lon[1], ra_upper_lon[2],
                               ra_upper_lon[3], ra_upper_lon[4],
                               np.flipud(az_right_lon[4]), np.flipud(ra_lower_lon[4]),
                               np.flipud(ra_lower_lon[3]), np.flipud(ra_lower_lon[2]),
                               np.flipud(ra_lower_lon[1]), np.flipud(ra_lower_lon[0]))
                            ).round(decimals=3)

        # apply 180 degree correction to longitude - code copied from
        # get_border_wkt...
        # TODO: simplify using np.mod?
        for ilon, llo in enumerate(lons):
            lons[ilon] = copysign(acos(cos(llo * pi / 180.)) / pi * 180,
                                  sin(llo * pi / 180.))

        lats = np.concatenate((az_left_lat[0], ra_upper_lat[0],
                               ra_upper_lat[1], ra_upper_lat[2],
                               ra_upper_lat[3], ra_upper_lat[4],
                               np.flipud(az_right_lat[4]), np.flipud(ra_lower_lat[4]),
                               np.flipud(ra_lower_lat[3]), np.flipud(ra_lower_lat[2]),
                               np.flipud(ra_lower_lat[1]), np.flipud(ra_lower_lat[0]))
                            ).round(decimals=2) # O(km)

        poly_border = ','.join(str(llo) + ' ' + str(lla) for llo, lla in zip(lons, lats))
        wkt = 'POLYGON((%s))' % poly_border
        new_geometry = WKTReader().read(wkt)

        # Get or create new geolocation of dataset
        # Returns False if it is the same as an already created one
        # (this may happen when a lot of data is processed)
        ds.geographic_location, cr = GeographicLocation.objects.get_or_create(
            geometry=new_geometry)
        connection.close()

        return ds, True

    def export2netcdf(self, n, ds, history_message='', filename='', all_bands=True):
        if not history_message:
            history_message = create_history_message(
                "sar_doppler.models.Dataset.objects.export2netcdf(n, ds, ",
                filename=filename)

        date_created = datetime.now(timezone.utc)

        drop_subswath_key = False
        if 'subswath' in n.get_metadata().keys():
            ii = int(n.get_metadata('subswath'))
            fn = nc_name(ds, ii)
            log_message = 'Exporting %s to %s (subswath %d)' % (n.filename, fn, ii+1)
        else:
            if not filename:
                raise ValueError('Please provide a netcdf filename!')
            fn = filename
            log_message = 'Exporting merged subswaths to %s' % fn
            # Metadata is the same except from the subswath attribute
            ii = 0
            drop_subswath_key = True

        original = Nansat(nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri),
                          subswath=ii)

        # Get metadata
        metadata = original.get_metadata()

        if drop_subswath_key:
            metadata.pop('subswath')

        metadata['title'] = n.get_metadata('title')
        metadata['title_no'] = n.get_metadata('title_no')
        metadata['summary'] = n.get_metadata('summary')
        metadata['summary_no'] = n.get_metadata('summary_no')

        def pretty_print_gcmd_keywords(kw):
            retval = ''
            value_prev = ''
            for key, value in kw.items():
                if value:
                    if value_prev:
                        retval += ' > '
                    retval += value
                    value_prev = value
            return retval

        # Set global metadata
        metadata['date_created'] = date_created.isoformat()
        metadata['date_created_type'] = 'Created'
        metadata['processing_level'] = 'Scientific'
        metadata['creator_role'] = 'Investigator'
        metadata['creator_type'] = 'person'
        metadata['creator_name'] = 'Morten Wergeland Hansen'
        metadata['creator_email'] = 'mortenwh@met.no'
        metadata['creator_institution'] = 'Norwegian Meteorological Institute (MET Norway)'

        metadata['contributor_name'] = 'Jeong-Won Park, Harald Johnsen, Geir Engen'
        metadata['contributor_role'] = 'Investigator, Investigator, Investigator'
        metadata['contributor_email'] = (
            'jeong-won.park@kopri.re.kr, hjoh@norceresearch.no, geen@norceresearch.no')
        metadata['contributor_institution'] = ('Korea Polar Research Institute (KOPRI), NORCE,'
            ' NORCE')

        metadata['project'] = (
                'Norwegian Space Agency project JOP.06.20.2: '
                'Reprocessing and analysis of historical data for '
                'future operationalization of Doppler shifts from '
                'SAR, NMI/ESA-NoR Envisat ASAR Doppler centroid shift'
                ' processing ID220131, ESA Prodex: Improved knowledge'
                ' of high latitude ocean circulation with Synthetic '
                'Aperture Radar (ISAR), ESA Prodex: Drift estimation '
                'of sea ice in the Arctic Ocean and sub-Arctic Seas '
                '(DESIce)'
            )
        metadata['publisher_type'] = 'institution'
        metadata['publisher_name'] = 'Norwegian Meteorological Institute'
        metadata['publisher_url'] = 'https://www.met.no/'
        metadata['publisher_email'] = 'data-management-group@met.no'

        metadata['doi'] = "https://doi.org/10.57780/esa-56fb232"

        metadata['dataset_production_status'] = 'Complete'

        # Get image boundary
        lon, lat= n.get_border()
        boundary = 'POLYGON (('
        for la, lo in list(zip(lat,lon)):
            boundary += '%.2f %.2f, '%(la,lo)
        boundary = boundary[:-2]+'))'
        # Set bounds as (lat,lon) following ACDD convention and EPSG:4326
        metadata['geospatial_bounds'] = boundary
        metadata['geospatial_bounds_crs'] = 'EPSG:4326'

        # Set software version
        metadata['sar_doppler'] = \
            subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(os.path.abspath(sar_doppler.__file__))).strip().decode()
        metadata['sar_doppler_resource'] = \
            "https://github.com/mortenwh/django-geo-spaas-sar-doppler"

        # history
        try:
            history = n.get_metadata('history')
        except ValueError:
            metadata['history'] = history_message
        else:
            metadata['history'] = history + '\n' + history_message

        # Set metadata from dict
        for key, val in metadata.items():
            n.set_metadata(key=key, value=val)

        bands = None
        # If all_bands=True, everything is exported. This is
        # useful when not all the bands in the list above have
        # been created
        #if not all_bands:
        #    # Bands to be exported
        #    bands = [
        #        n.get_band_number("incidence_angle"),
        #        n.get_band_number("sensor_view_corrected"),
        #        n.get_band_number("sensor_azimuth"),
        #        n.get_band_number("topographic_height"),
        #        n.get_band_number({"standard_name":
        #            "surface_backwards_doppler_centroid_frequency_shift_of_radar_wave"}),
        #        n.get_band_number({"standard_name":
        #            "standard_deviation_of_surface_backwards_doppler_centroid_frequency_"
        #            "shift_of_radar_wave"}),
        #        n.get_band_number("fe"),
        #        n.get_band_number("fgeo"),
        #        n.get_band_number("fdg"),
        #        n.get_band_number("fww"),
        #        n.get_band_number("std_fww"),
        #        n.get_band_number("Ur"),
        #        n.get_band_number("std_Ur"),
        #        # Needed for intermediate calculations
        #        n.get_band_number("U3g_0"),
        #        n.get_band_number("U3g_1"),
        #        n.get_band_number("U3g_2"),
        #        n.get_band_number("dcp0"),
        #        n.get_band_number({
        #            'standard_name': 'surface_backwards_scattering_coefficient_of_radar_wave'}),
        #        # Valid pixels
        #        n.get_band_number("valid_land_doppler"),
        #        n.get_band_number("valid_sea_doppler"),
        #        n.get_band_number("valid_doppler"),
        #    ]
        # Export data to netcdf
        logging.debug(log_message)
        n.export(filename=fn) #, bands=bands)

        # Nansat has filename metadata, which is wrong, and adds GCPs as variables.
        # Just remove everything.
        nc = netCDF4.Dataset(fn, 'a')
        if 'filename' in nc.ncattrs():
            nc.delncattr('filename')
            tmp = nc.variables.pop("GCPX", "")
            tmp = nc.variables.pop("GCPY", "")
            tmp = nc.variables.pop("GCPZ", "")
            tmp = nc.variables.pop("GCPPixel", "")
            tmp = nc.variables.pop("GCPLine", "")

        # Nansat adds units to the lon/lat grids but they are wrong
        # ("deg N" should be "degrees_north")
        nc["latitude"].units = "degrees_north"
        # ("deg E" should be "degrees_east")
        nc["longitude"].units = "degrees_east"

        nc.close()

        # Add netcdf uri to DatasetURIs
        ncuri = 'file://localhost' + fn

        locked = True
        while locked:
            try:
                new_uri, created = DatasetURI.objects.get_or_create(uri=ncuri, dataset=ds)
            except OperationalError as oe:
                locked = True
            else:
                locked = False
        connection.close()

        return new_uri, created

    def process(self, ds, force=False, *args, **kwargs):
        """ Create data products

        Returns
        =======
        ds : geospaas.catalog.models.Dataset
        processed : Boolean
            Flag to indicate if the dataset was processed or not
        """
        history_message = create_history_message(
                "sar_doppler.models.Dataset.objects.process(ds, ",
                force=force, *args, **kwargs)

        swath_data = {}

        # Set media path (where images will be stored)
        mp = media_path(
                module_name(),
                nansat_filename(ds.dataseturi_set.get(uri__endswith = '.gsar').uri)
            )

        # Read subswaths 
        dss = {1: None, 2: None, 3: None, 4: None, 5: None}
        processed = [True, True, True, True, True]
        failing = [False, False, False, False, False]
        for i in range(self.N_SUBSWATHS):
            # Check if the data has already been processed
            try:
                fn = nansat_filename(ds.dataseturi_set.get(uri__endswith='swath%d.nc'%i).uri)
            except DatasetURI.DoesNotExist:
                processed[i] = False
            else:
                dd = Nansat(fn)
                try:
                    std_Ur = dd['std_Ur']
                except ValueError:
                    processed[i] = False
            if processed[i] and not force:
                continue
            # Process from scratch to avoid duplication of bands
            fn = nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri)
            try:
                dd = Doppler(fn, subswath=i)
            except Exception as e:
                logging.error('%s (Filename, subswath [1-5]): (%s, %d)' % (str(e), fn, i+1))
                failing[i] = True
                continue

            # Check if the file is corrupted
            try:
                inc = dd['incidence_angle']
            except Exception as e:
                logging.error('%s (Filename, subswath [1-5]): (%s, %d)' % (str(e), fn, i+1))
                failing[i] = True
                continue

            dss[i+1] = dd

        if all(processed) and not force:
            logging.debug("%s: The dataset has already been processed." % nansat_filename(
                ds.dataseturi_set.get(uri__endswith='.gsar').uri))
            return ds, False

        if all(failing):
            logging.error("Processing of all subswaths is failing: %s" % nansat_filename(
                ds.dataseturi_set.get(uri__endswith='.gsar').uri))
            return ds, False

        if any(failing):
            logging.error("Some but not all subswaths processed: %s" % nansat_filename(
                ds.dataseturi_set.get(uri__endswith='.gsar').uri))
            return ds, False

        logging.debug("Processing %s" % nansat_filename(
            ds.dataseturi_set.get(uri__endswith='.gsar').uri))

        # Loop subswaths, process each of them
        processed = False

        # Find wind
        db_locked = True
        while db_locked:
            try:
                wind_fn = nansat_filename(
                    Dataset.objects.get(
                        source__platform__short_name = 'ERA15DAS',
                        time_coverage_start__lte = ds.time_coverage_end,
                        time_coverage_end__gte = ds.time_coverage_start
                    ).dataseturi_set.get().uri
                )
            except OperationalError as oe:
                db_locked = True
            except Exception as e:
                logging.error("%s - in search for ERA15DAS data (%s, %s, %s) " % (
                    str(e),
                    nansat_filename(ds.dataseturi_set.get(uri__endswith=".gsar").uri),
                    ds.time_coverage_start,
                    ds.time_coverage_end
                ))
                return ds, False
            else:
                db_locked = False
        connection.close()

	# Get range bias corrected Doppler
        fdg = {}
        offset_corrected = {}
        offset = {}
        fdg[1], offset_corrected[1], offset[1] = dss[1].geophysical_doppler_shift(wind=wind_fn)
        fdg[2], offset_corrected[2], offset[2] = dss[2].geophysical_doppler_shift(wind=wind_fn)
        fdg[3], offset_corrected[3], offset[3] = dss[3].geophysical_doppler_shift(wind=wind_fn)
        fdg[4], offset_corrected[4], offset[4] = dss[4].geophysical_doppler_shift(wind=wind_fn)
        fdg[5], offset_corrected[5], offset[5] = dss[5].geophysical_doppler_shift(wind=wind_fn)

        def redo_offset_corr(ff, corr, old_offset, new_offset):
            """ If a subswath has not been corrected by land
            reference, but another one has, this function
            replaces the cdop estimated offset with a new one.
            """
            if not corr:
                ff += old_offset
                ff -= new_offset
            return ff, True, new_offset

        # Find the mean offset from those subswaths that have been
        # offset corrected with land reference
        count = 0
        sum_offsets = 0
        for key in offset_corrected.keys():
            if offset_corrected[key]:
                count += 1
                sum_offsets += offset[key]

        # If any subswaths have been corrected with land reference,
        # redo the offset correction for any subswaths that have not
        # been offset corrected with land reference
        if sum_offsets > 0:
            new_offset = sum_offsets/count
            for key in offset_corrected.keys():
                fdg[key], offset_corrected[key], offset[key] = redo_offset_corr(fdg[key],
                    offset_corrected[key], offset[key], new_offset)
            offset_corrected['all'] = True

        if 'all' not in offset_corrected.keys():
            offset_corrected['all'] = False

        """ This looks nice but is risky if border values are wrong..
        def get_overlap(d1, d2):
            b1 = d1.get_border_geometry()
            b2 = d2.get_border_geometry()
            intersection = b1.Intersection(b2)
            lo1,la1 = d1.get_geolocation_grids()
            overlap = np.zeros(lo1.shape)
            for i in range(lo1.shape[0]):
                for j in range(lo1.shape[1]):
                    wkt_point = 'POINT(%.5f %.5f)' % (lo1[i,j], la1[i,j])
                    overlap[i,j] = intersection.Contains(ogr.CreateGeometryFromWkt(wkt_point))

            return overlap

        logging.debug("%s" % nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri))
        # Find pixels in dss[1] which overlap with pixels in dss[2]
        overlap12 = get_overlap(dss[1], dss[2])
        # Find pixels in dss[2] which overlap with pixels in dss[1]
        overlap21 = get_overlap(dss[2], dss[1])
        # and so on..
        overlap23 = get_overlap(dss[2], dss[3])
        overlap32 = get_overlap(dss[3], dss[2])
        overlap34 = get_overlap(dss[3], dss[4])
        overlap43 = get_overlap(dss[4], dss[3])
        overlap45 = get_overlap(dss[4], dss[5])
        overlap54 = get_overlap(dss[5], dss[4])

        # Get median values at overlapping borders
        median12 = np.nanmedian(fdg[1][np.where(overlap12)])
        median21 = np.nanmedian(fdg[2][np.where(overlap21)])
        median23 = np.nanmedian(fdg[2][np.where(overlap23)])
        median32 = np.nanmedian(fdg[3][np.where(overlap32)])
        median34 = np.nanmedian(fdg[3][np.where(overlap34)])
        median43 = np.nanmedian(fdg[4][np.where(overlap43)])
        median45 = np.nanmedian(fdg[4][np.where(overlap45)])
        median54 = np.nanmedian(fdg[5][np.where(overlap54)])


        # Adjust levels to align at subswath borders
        if offset_corrected[1] and offset_corrected[2]:
            offset_corrected['all'] = True
            fdg[1] -= median12 - np.nanmedian(np.array([median12, median21]))
        elif offset_corrected[2]:
            offset_corrected['all'] = True
            fdg[1] -= median12 - median21
        if offset_corrected[1] and offset_corrected[2]:
            fdg[2] -= median21 - np.nanmedian(np.array([median12, median21]))
        elif offset_corrected[1]:
            offset_corrected['all'] = True
            fdg[2] -= median21 - median12

        if offset_corrected[2] and offset_corrected[3]:
            offset_corrected['all'] = True
            fdg[1] -= median23 - np.nanmedian(np.array([median23, median32]))
            fdg[2] -= median23 - np.nanmedian(np.array([median23, median32]))
        elif offset_corrected[3]:
            offset_corrected['all'] = True
            fdg[1] -= median23 - median32
            fdg[2] -= median23 - median32
        if offset_corrected[2] and offset_corrected[3]:
            fdg[3] -= median32 - np.nanmedian(np.array([median23, median32]))
        elif offset_corrected[2]:
            offset_corrected['all'] = True
            fdg[3] -= median32 - median23

        if offset_corrected[3] and offset_corrected[4]:
            offset_corrected['all'] = True
            fdg[1] -= median34 - np.nanmedian(np.array([median34, median43]))
            fdg[2] -= median34 - np.nanmedian(np.array([median34, median43]))
            fdg[3] -= median34 - np.nanmedian(np.array([median34, median43]))
        elif offset_corrected[4]:
            offset_corrected['all'] = True
            fdg[1] -= median34 - median43
            fdg[2] -= median34 - median43
            fdg[3] -= median34 - median43
        if offset_corrected[3] and offset_corrected[4]:
            fdg[4] -= median43 - np.nanmedian(np.array([median34, median43]))
        elif offset_corrected[3]:
            offset_corrected['all'] = True
            fdg[4] -= median43 - median34

        if offset_corrected[4] and offset_corrected[5]:
            offset_corrected['all'] = True
            fdg[1] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[2] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[3] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[4] -= median45 - np.nanmedian(np.array([median45, median54]))
        elif offset_corrected[5]:
            offset_corrected['all'] = True
            fdg[1] -= median45 - median54
            fdg[2] -= median45 - median54
            fdg[3] -= median45 - median54
            fdg[4] -= median45 - median54
        if offset_corrected[4] and offset_corrected[5]:
            fdg[5] -= median54 - np.nanmedian(np.array([median45, median54]))
        elif offset_corrected[4]:
            offset_corrected['all'] = True
            fdg[5] -= median54 - median45

        if not offset_corrected['all']:
            # Just align all
            fdg[1] -= median12 - np.nanmedian(np.array([median12, median21]))
            fdg[2] -= median21 - np.nanmedian(np.array([median12, median21]))
            fdg[1] -= median23 - np.nanmedian(np.array([median23, median32]))
            fdg[2] -= median23 - np.nanmedian(np.array([median23, median32]))
            fdg[3] -= median32 - np.nanmedian(np.array([median23, median32]))
            fdg[1] -= median34 - np.nanmedian(np.array([median34, median43]))
            fdg[2] -= median34 - np.nanmedian(np.array([median34, median43]))
            fdg[3] -= median34 - np.nanmedian(np.array([median34, median43]))
            fdg[4] -= median43 - np.nanmedian(np.array([median34, median43]))
            fdg[1] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[2] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[3] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[4] -= median45 - np.nanmedian(np.array([median45, median54]))
            fdg[5] -= median54 - np.nanmedian(np.array([median45, median54]))
        """

        nc_uris = []
        for key in dss.keys():
            # Add electronic Doppler as band
            dss[key].add_band(
                array=dss[key].range_bias(),
                parameters={
                    "name": "fe",
                    "long_name": "Doppler frequency shift due to electronic mispointing",
                    "units": "Hz",
                }
            )

            # Add geometric Doppler as band
            dss[key].add_band(
                array=dss[key].predicted(),
                parameters={
                    "name": "fgeo",
                    "long_name": "Doppler frequency shift due to orbit geometry",
                    "units": "Hz",
                }
            )

            # Add fdg[key] as band
            dss[key].add_band(
                array=fdg[key],
                parameters={
                    "name": "fdg",
                    "long_name": "Radar Doppler frequency shift due to surface velocity",
                    "units": "Hz",
                    "offset_corrected": str(offset_corrected["all"]),
                }
            )

            # Add wind information as bands
            fww, dfww, u10, phi = dss[key].wind_waves_doppler(wind_fn)

            dss[key].add_band(
                array = u10,
                parameters = {
                    "name": "wind_speed",
                    "standard_name": "wind_speed",
                    "long_name": "Wind speed used in CDOP calculation",
                    "units": "m s-1",
                    "file": wind_fn,
                }
            )
            dss[key].add_band(
                array = phi,
                parameters = {
                    "name": "wind_direction",
                    "long_name": "SAR look relative wind from direction used in CDOP calculation",
                    "units": "degree",
                    "file": wind_fn,
                }
            )
            dss[key].add_band(
                array = fww,
                parameters = {
                    "name": "fww",
                    "long_name": "Radar Doppler frequency shift due to wind waves",
                    "units": "Hz",
                }
            )

            dss[key].add_band(
                array = dfww,
                parameters = {
                    "name": "std_fww",
                    "long_name": ("Standard deviation of radar Doppler frequency shift due"
                                  " to wind waves"),
                    "units": "Hz",
                }
            )

            # Calculate range current velocity component
            v_current, std_v, offset_corrected_tmp = \
                dss[key].surface_radial_doppler_sea_water_velocity(wind_fn, fdg=fdg[key],
                    offset_corrected=offset_corrected['all'])
            wkv = 'surface_radial_doppler_sea_water_velocity'
            dss[key].add_band(
                array = v_current,
                parameters = {
                    "name": "u_range",
                    "long_name": "Sea surface current velocity in range direction",
                    "units": "m s-1",
                    "offset_corrected": str(offset_corrected["all"]),
                }
            )

            dss[key].add_band(
                array=std_v,
                parameters={
                    "name": "std_u_range",
                    "long_name": ("Standard deviation of sea surface current velocity in range"
                                  " direction"),
                    "units": "m s-1",
                })
  
            # Set satellite pass
            lon, lat = dss[key].get_geolocation_grids()
            gg = np.gradient(lat, axis=0)
            dss[key].add_band(
                array = gg,
                parameters = {
                    'name': 'sat_pass',
                    "long_name": "satellite pass",
                    'comment': 'ascending pass is >0, descending pass is <0'
                }
            )

            title = (
                'Calibrated geophysical %s %s wide-swath range '
                'Doppler frequency shift retrievals in %s '
                'polarisation, subswath %s, %s') %(
                        pti.get_gcmd_platform('envisat')['Short_Name'],
                        pti.get_gcmd_instrument('asar')['Short_Name'],
                        ds.sardopplerextrametadata_set.get().polarization,
                        key,
                        dss[key].get_metadata('time_coverage_start')
                )
            dss[key].set_metadata(key='title', value=title)
            title_no = (
                'Kalibrert geofysisk %s %s Dopplerskift i satellitsveip %s og %s polarisering, %s'
            ) %(
                    pti.get_gcmd_platform('envisat')['Short_Name'],
                    pti.get_gcmd_instrument('asar')['Short_Name'],
                    key,
                    ds.sardopplerextrametadata_set.get().polarization,
                    dss[key].get_metadata('time_coverage_start')
            )
            dss[key].set_metadata(key='title_no', value=title_no)
            summary = (
                "Calibrated geophysical range Doppler frequency shift "
                "retrievals from an %s %s wide-swath acqusition "
                "obtained on %s. The geophysical Doppler shift "
                "depends on the ocean wave-state and the sea surface "
                "current. In the absence of current, the geophysical "
                "Doppler shift is mostly related to the local wind "
                "speed and direction. The present dataset is in %s "
                "polarization, sub-swath %s.") % (
                    pti.get_gcmd_platform('envisat')['Short_Name'],
                    pti.get_gcmd_instrument('asar')['Short_Name'],
                    dss[key].get_metadata('time_coverage_start'),
                    ds.sardopplerextrametadata_set.get().polarization,
                    key
                )
            dss[key].set_metadata(key='summary', value=summary)
            summary_no = (
                "Kalibrert geofysisk Dopplerskift fra %s %s målt %s. "
                "Det geofysiske Dopplerskiftet avhenger av "
                "havbølgetilstand og overflatestrøm. Ved fravær av "
                "strøm er det geofysiske Dopplerskiftet stort sett "
                "relatert til den lokale vindhastigheten og dens "
                "retning. Foreliggende datasett representerer "
                "satellittsveip %s og %s polarisering.") % (
                    pti.get_gcmd_platform('envisat')['Short_Name'],
                    pti.get_gcmd_instrument('asar')['Short_Name'],
                    dss[key].get_metadata('time_coverage_start'),
                    key,
                    ds.sardopplerextrametadata_set.get().polarization
                )
            dss[key].set_metadata(key='summary_no', value=summary_no)

            subswathno = key-1
            calibration_ds = Dataset.objects.get(
                dataseturi__uri__contains=dss[key].get_lut_filename())

            new_uri, created = self.export2netcdf(dss[key], ds, history_message=history_message)
            nc_uris.append(new_uri)
            processed = True

        # Set parent dataset ID
        for uri in nc_uris:
            related_datasets = "no.met:3df54118-e9d8-4fe4-a773-e4c2cb35c125 (parent)"
            ncd = netCDF4.Dataset(nansat_filename(uri.uri), "a")
            ncd.related_dataset = related_datasets
            ncd.close()

        # Merge subswaths
        m, nc_uri = self.merge_swaths(ds, **kwargs)
        # Create MMD file
        create_mmd_file(nansat_filename(calibration_ds.dataseturi_set.get().uri), nc_uri)

        return ds, processed

    def merge_swaths(self, ds, **kwargs):
        """Create Nansat object with merged swaths, export to netcdf,
        and add uri.
        """
        m = create_merged_swaths(ds)
        # Add file to db
        new_uri, created = self.export2netcdf(m, ds, filename=m.filename)
        connection.close()
        uri = ds.dataseturi_set.get(uri__contains='merged')
        connection.close()
        return m, uri

    def get_merged_swaths(self, ds, reprocess=False, **kwargs):
        """Get merged swaths
        """
        def get_uri(ds):
            """ Get uri of merged swaths file. Return empty string if
            the uri doesn't exist.
            """
            try:
                uri = ds.dataseturi_set.get(uri__contains='merged')
            except DatasetURI.DoesNotExist:
                uri = ""
            return uri
        uri = get_uri(ds)
        if reprocess or not uri:
            n = Nansat(nansat_filename(ds.dataseturi_set.get(uri__contains='subswath1').uri))
            if not n.has_band('Ur') or reprocess:
                # Process dataset
                ds, processed = self.process(ds, force=True, **kwargs)
            m, uri = self.merge_swaths(ds)
        else:
            m = Nansat(nansat_filename(uri.uri))

        return m, uri

    def imshow_fdg(self, ds, png_fn, title=None, **kwargs):
        """Plot geophysical doppler shift
        """
        merged = self.get_merged_swaths(ds, **kwargs)
        # Actual plotting
        lon, lat = merged.get_geolocation_grids()
        globe = ccrs.Globe(ellipse='WGS84', semimajor_axis=6378137, flattening=1/298.2572235604902)
        proj = ccrs.Stereographic(
                central_longitude=np.mean(lon),
                central_latitude=np.mean(lat),
                globe=globe
            )

        fig, axs = plt.subplots(1, 1, subplot_kw={'projection': proj}, figsize=(20, 20))
        extent = [np.min(lon)-.5, np.max(lon)+.5, np.min(lat)-.5, np.max(lat)+.5]
        land_f = cfeature.NaturalEarthFeature(
                'physical', 'land', '50m', edgecolor='face', facecolor='lightgray'
            )

        axs.set_extent(extent, crs=ccrs.PlateCarree())
        # THIS IS FASTER THAN contourf BUT I CAN'T GET IT CORRECTLY...
        # im = axs.imshow(
        #         np.flipud(s0),
        #         extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)],
        #         transform=ccrs.PlateCarree(),
        #         cmap='gray',
        #         clim=[-20,0],
        #         interpolation=None
        #     )
        axs.gridlines(color='gray', linestyle='--')
        #axs.add_feature(land_f)
        axs.coastlines(resolution='50m')
        im = axs.contourf(
                lon,
                lat,
                merged['fdg'],
                400,
                vmin = -60,
                vmax = 60,
                transform = ccrs.PlateCarree(),
                cmap = cmocean.cm.balance
            )
        plt.colorbar(im)
        if title:
            axs.set_title(title, y=1.05, fontsize=20)

        plt.savefig(png_fn)

        return merged






    #def bayesian_wind(self):
    #    # Find matching NCEP forecast wind field
    #    wind = [] # do not do any wind correction now, since we have lookup tables
    #    wind = Dataset.objects.filter(
    #            source__platform__short_name='NCEP-GFS',
    #            time_coverage_start__range=[
    #                parse(swath_data[i].get_metadata()['time_coverage_start'])
    #                - timedelta(hours=3),
    #                parse(swath_data[i].get_metadata()['time_coverage_start'])
    #                + timedelta(hours=3)
    #            ]
    #        )
    #    if wind:
    #        dates = [w.time_coverage_start for w in wind]
    #        # TODO: Come back later (!!!)
    #        nearest_date = min(dates, key=lambda d:
    #                abs(d-parse(swath_data[i].get_metadata()['time_coverage_start']).replace(tzinfo=timezone.utc)))
    #        fww = swath_data[i].wind_waves_doppler(
    #                nansat_filename(wind[dates.index(nearest_date)].dataseturi_set.all()[0].uri),
    #                pol
    #            )

    #        swath_data[i].add_band(array=fww, parameters={
    #            'wkv':
    #            'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_wind_waves'
    #        })

    #        fdg = swath_data[i].geophysical_doppler_shift(
    #            wind=nansat_filename(wind[dates.index(nearest_date)].dataseturi_set.all()[0].uri)
    #        )

    #        # Estimate current by subtracting wind-waves Doppler
    #        theta = swath_data[i]['incidence_angle'] * np.pi / 180.
    #        vcurrent = -np.pi * (fdg - fww) / (112. * np.sin(theta))

    #        # Smooth...
    #        # vcurrent = median_filter(vcurrent, size=(3,3))
    #        swath_data[i].add_band(
    #            array=vcurrent,
    #            parameters={
    #                'wkv': 'surface_radial_doppler_sea_water_velocity'
    #            })

