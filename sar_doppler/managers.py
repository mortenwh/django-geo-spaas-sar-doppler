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
from sar_doppler.utils import create_mmd_files


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

    def export2netcdf(self, n, ds, history_message='', filename='', all_bands=False):
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

        # Get and set dataset id
        if 'id' not in n.get_metadata().keys():
            if os.path.isfile(fn):
                tmp = Nansat(fn)
                if 'id' in tmp.get_metadata().keys():
                    n.set_metadata(key='id', value=tmp.get_metadata('id'))
                else:
                    n.set_metadata(key='id', value=str(uuid.uuid4()))
            else:
                n.set_metadata(key='id', value=str(uuid.uuid4()))
        # Set global metadata
        metadata['date_created'] = date_created.isoformat()
        metadata['date_created_type'] = 'Created'
        metadata['processing_level'] = 'Scientific'
        metadata['creator_role'] = 'Investigator'
        metadata['creator_type'] = 'person'
        metadata['creator_name'] = 'Morten Wergeland Hansen'
        metadata['creator_email'] = 'mortenwh@met.no'
        metadata['creator_institution'] = 'Norwegian Meteorological Institute'

        metadata['contributor_name'] = 'Jeong-Won Park, Geir Engen, Harald Johnsen'
        metadata['contributor_role'] = 'Investigator, Investigator, Investigator'
        metadata['contributor_email'] = (
            'jeong-won.park@kopri.re.kr, geen@norceresearch.no, hjoh@norceresearch.no')
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
        metadata['publisher_email'] = 'csw-services@met.no'

        metadata['references'] = "https://data.met.no/dataset/%s(Dataset landing page)" % \
            n.get_metadata("id")
        metadata['doi'] = "10.57780/esa-56fb232"

        metadata['dataset_production_status'] = 'Complete'

        # Get image boundary
        lon,lat= n.get_border()
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
        if not all_bands:
            # Bands to be exported
            bands = [
                n.get_band_number("incidence_angle"),
                n.get_band_number("sensor_view_corrected"),
                n.get_band_number("sensor_azimuth"),
                n.get_band_number("topographic_height"),
                n.get_band_number({"standard_name":
                    "surface_backwards_doppler_centroid_frequency_shift_of_radar_wave"}),
                n.get_band_number({"standard_name":
                    "standard_deviation_of_surface_backwards_doppler_centroid_frequency_"
                    "shift_of_radar_wave"}),
                n.get_band_number("fe"),
                n.get_band_number("fgeo"),
                n.get_band_number("fdg"),
                n.get_band_number("fww"),
                n.get_band_number("std_fww"),
                n.get_band_number("Ur"),
                n.get_band_number("std_Ur"),
                # Needed for intermediate calculations
                n.get_band_number("U3g_0"),
                n.get_band_number("U3g_1"),
                n.get_band_number("U3g_2"),
                n.get_band_number("dcp0"),
                n.get_band_number({
                    'standard_name': 'surface_backwards_scattering_coefficient_of_radar_wave'}),
                # Valid pixels
                n.get_band_number("valid_land_doppler"),
                n.get_band_number("valid_sea_doppler"),
                n.get_band_number("valid_doppler"),
            ]
        # Export data to netcdf
        logging.debug(log_message)
        n.export(filename=fn, bands=bands)

        # Nansat has filename metadata, which is wrong, and adds GCPs as variables.
        # Just remove everything.
        nc = netCDF4.Dataset(fn, 'a')
        if 'filename' in nc.ncattrs():
            nc.delncattr('filename')
            tmp = nc.variables.pop("GCPX")
            tmp = nc.variables.pop("GCPY")
            tmp = nc.variables.pop("GCPZ")
            tmp = nc.variables.pop("GCPPixel")
            tmp = nc.variables.pop("GCPLine")
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
                fn = nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%i).uri)
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
            wkv = 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_' \
                'velocity'
            dss[key].add_band(
                array=fdg[key],
                parameters={
                    'wkv': wkv,
                    'offset_corrected': str(offset_corrected['all']),
                    'valid_min': -200, 
                    'valid_max': 200,
                }
            )

            # Add wind doppler and its uncertainty as bands
            fww, dfww = dss[key].wind_waves_doppler(wind_fn)
            wkv = 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_wind_waves'
            dss[key].add_band(
                array = fww,
                parameters = {
                    'wkv': wkv,
                    'valid_min': -200, 
                    'valid_max': 200,
                }
            )

            dss[key].add_band(
                array = dfww,
                parameters = {
                    'name': 'std_fww',
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
                    'wkv': wkv,
                    'offset_corrected': str(offset_corrected['all']),
                    'valid_min': -5,
                    'valid_max': 5,
                }
            )

            dss[key].add_band(
                array=std_v,
                parameters={
                    'name': 'std_Ur',
                })
  
            # Set satellite pass
            lon,lat = dss[key].get_geolocation_grids()
            gg = np.gradient(lat, axis=0)
            dss[key].add_band(
                array = gg,
                parameters = {
                    'name': 'sat_pass',
                    "long_name": "satellite pass",
                    'comment': 'ascending pass is >0, descending pass is <0'
                }
            )

            title = title = (
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
            summary = summary = (
                'Calibrated geophysical %s %s wide-swath range Doppler frequency shift '
                'retrievals in %s polarization, sub-swath %s. The '
                'data was acquired on %s.') % (
                    pti.get_gcmd_platform('envisat')['Short_Name'],
                    pti.get_gcmd_instrument('asar')['Short_Name'],
                    ds.sardopplerextrametadata_set.get().polarization,
                    key,
                    dss[key].get_metadata('time_coverage_start')
                )
            dss[key].set_metadata(key='summary', value=summary)
            summary_no = (
                'Kalibrert geofysisk %s %s Dopplerskift i satellittsveip %s og '
                '%s polarisering. Dataene ble samlet %s.') % (
                    pti.get_gcmd_platform('envisat')['Short_Name'],
                    pti.get_gcmd_instrument('asar')['Short_Name'],
                    key,
                    ds.sardopplerextrametadata_set.get().polarization,
                    dss[key].get_metadata('time_coverage_start')
                )
            dss[key].set_metadata(key='summary_no', value=summary_no)

            subswathno = key-1
            calibration_ds = Dataset.objects.get(
                dataseturi__uri__contains=dss[key].get_lut_filename())

            """ The gsar dataset stored in geospaas is not stored as
            netcdf. Instead, all subswaths are regarded as separate
            datasets. The relations are added below after creating the
            netcdf files."""
            #dss[key].set_metadata(
            #        key = 'related_dataset_id',
            #        value = calibration_ds.entry_id
            #    )
            #dss[key].set_metadata(
            #        key = 'related_dataset_relation_type',
            #        value = 'auxiliary'
            #    )

            new_uri, created = self.export2netcdf(dss[key], ds, history_message=history_message)
            nc_uris.append(new_uri)
            processed = True

        # Dette virker ikke ved polene - blir minneproblem! Must find
        # en annen metode to sette samme subswaths
        #m = self.create_merged_swaths(ds)

        # Set auxiliary related_dataset IDs
        aux_datasets = {}
        # Get all dataset IDs
        for uri in nc_uris:
            ncd = netCDF4.Dataset(nansat_filename(uri.uri))
            aux_datasets[nansat_filename(uri.uri)] = "%s:%s (auxiliary)" % (ncd.naming_authority,
                                                                            ncd.id)
            ncd.close()

        # Set related dataset IDs
        for uri in nc_uris:
            related_datasets = "no.met:3df54118-e9d8-4fe4-a773-e4c2cb35c125 (parent)"
            others = aux_datasets.copy()
            others.pop(nansat_filename(uri.uri))
            for key in others:
                related_datasets += ", %s" % others[key]
            ncd = netCDF4.Dataset(nansat_filename(uri.uri), "a")
            ncd.related_dataset = related_datasets
            ncd.close()

        # Create MMD files
        create_mmd_files(nansat_filename(calibration_ds.dataseturi_set.get().uri), nc_uris)

        return ds, processed

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
            m = self.create_merged_swaths(ds)
            uri = ds.dataseturi_set.get(uri__contains='merged')
        connection.close()
        m = Nansat(nansat_filename(uri.uri))

        return m

    def create_merged_swaths(self, ds, EPSG = 4326, **kwargs):
        """Merge swaths, add dataseturi, and return Nansat object.

        EPSG options:
            - 4326: WGS 84 / longlat
            - 3995: WGS 84 / Arctic Polar Stereographic
        """
        history_message = create_history_message(
                "sar_doppler.models.Dataset.objects.create_merged_swaths(ds, ",
                EPSG=EPSG, **kwargs)
        nn = {}
        nn[0] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%0).uri))
        lon0, lat0 = nn[0].get_geolocation_grids()
        nn[1] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%1).uri))
        lon1, lat1 = nn[1].get_geolocation_grids()
        nn[2] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%2).uri))
        lon2, lat2 = nn[2].get_geolocation_grids()
        nn[3] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%3).uri))
        lon3, lat3 = nn[3].get_geolocation_grids()
        nn[4] = Doppler(nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%4).uri))
        lon4, lat4 = nn[4].get_geolocation_grids()

        connection.close()

        pol = nn[0].get_metadata('polarization')

        dlon = np.mean([
                        np.abs(np.mean(np.gradient(lon0, axis=1))),
                        np.abs(np.mean(np.gradient(lon1, axis=1))),
                        np.abs(np.mean(np.gradient(lon2, axis=1))),
                        np.abs(np.mean(np.gradient(lon3, axis=1))),
                        np.abs(np.mean(np.gradient(lon4, axis=1)))
                    ])
        nx = len(np.arange(
                    np.array([
                        lon0.min(),
                        lon1.min(),
                        lon2.min(),
                        lon3.min(),
                        lon4.min()]).min(),
                    np.array([
                        lon0.max(),
                        lon1.max(),
                        lon2.max(),
                        lon3.max(),
                        lon4.max()]).max(),
                    dlon))
        dlat = np.mean([
                        np.abs(np.mean(np.gradient(lat0, axis=0))),
                        np.abs(np.mean(np.gradient(lat1, axis=0))),
                        np.abs(np.mean(np.gradient(lat2, axis=0))),
                        np.abs(np.mean(np.gradient(lat3, axis=0))),
                        np.abs(np.mean(np.gradient(lat4, axis=0)))
                    ])
        ny = len(np.arange(
                    np.array([
                        lat0.min(),
                        lat1.min(),
                        lat2.min(),
                        lat3.min(),
                        lat4.min()]).min(),
                    np.array([
                        lat0.max(),
                        lat1.max(),
                        lat2.max(),
                        lat3.max(),
                        lat4.max()]).max(),
                    dlat))

        if ny is None:
            ny = np.array([
                nn[0].shape()[0],
                nn[1].shape()[0],
                nn[2].shape()[0],
                nn[3].shape()[0],
                nn[4].shape()[0]
            ]).max()

        ## DETTE VIRKER IKKE..
        #sensor_view = np.sort(
        #        np.append(np.append(np.append(np.append(
        #            nn[0]['sensor_view'][0,:],
        #            nn[1]['sensor_view'][0,:]),
        #            nn[2]['sensor_view'][0,:]), 
        #            nn[3]['sensor_view'][0,:]),
        #            nn[4]['sensor_view'][0,:]))

        #nx = sensor_view.size
        #x = np.arange(nx)

        #def func(x, a, b, c, d):
        #    return a*x**3+b*x**2+c*x+d

        #def linear_func(x, a, b):
        #    return a*x + b

        #azimuth_time = np.sort(
        #        np.append(np.append(np.append(np.append(
        #            nn[0].get_azimuth_time(),
        #            nn[1].get_azimuth_time()),
        #            nn[2].get_azimuth_time()),
        #            nn[3].get_azimuth_time()),
        #            nn[4].get_azimuth_time()))
        #dt = azimuth_time.max() - azimuth_time[0]
        #tt = np.arange(0, dt, dt/ny)
        #tt = np.append(np.array([-dt/ny], dtype='<m8[us]'), tt)
        #tt = np.append(tt, tt[-1]+np.array([dt/ny, 2*dt/ny], dtype='<m8[us]'))
        #ny = len(tt)

        ## AZIMUTH_TIME
        #azimuth_time = (np.datetime64(azimuth_time[0])+tt).astype(datetime)

        #popt, pcov = curve_fit(func, x, sensor_view)
        ## SENSOR VIEW ANGLE
        #alpha = np.ones((ny, sensor_view.size))*np.deg2rad(func(x, *popt))

        #range_time = np.sort(
        #        np.append(np.append(np.append(np.append(
        #            nn[0].get_range_time(),
        #            nn[1].get_range_time()),
        #            nn[2].get_range_time()),
        #            nn[3].get_range_time()),
        #            nn[4].get_range_time()))
        #popt, pcov = curve_fit(linear_func, x, range_time)
        ## RANGE_TIME
        #range_time = linear_func(x, *popt)

        #ecefPos, ecefVel = Doppler.orbital_state_vectors(azimuth_time)
        #eciPos, eciVel = ecef2eci(ecefPos, ecefVel, azimuth_time)

        ## Get satellite hour angle
        #satHourAng = np.deg2rad(Doppler.satellite_hour_angle(azimuth_time, ecefPos, ecefVel))

        ## Get attitude from the Envisat yaw steering law
        #psi, gamma, phi = np.deg2rad(Doppler.orbital_attitude_vectors(azimuth_time, satHourAng))

        #U1, AX1, S1 = Doppler.step_one_calculations(alpha, psi, gamma, phi, eciPos)
        #S2, U2, AX2 = Doppler.step_two_calculations(satHourAng, S1, U1, AX1)
        #S3, U3, AX3 = Doppler.step_three_a_calculations(eciPos, eciVel, S2, U2, AX2)
        #U3g = Doppler.step_three_b_calculations(S3, U3, AX3)

        #P3, U3g, lookAng = Doppler.step_four_calculations(S3, U3g, AX3, range_time)
        #dcm = dcmeci2ecef(azimuth_time, 'IAU-2000/2006')
        #lat = np.zeros((ny, nx))
        #lon = np.zeros((ny, nx))
        #alt = np.zeros((ny, nx))
        #for i in range(P3.shape[1]):
        #    ecefPos = np.matmul(dcm[0], P3[:,i,:,0, np.newaxis])
        #    lla = ecef2lla(ecefPos)
        #    lat[:,i] = lla[:,0]
        #    lon[:,i] = lla[:,1]
        #    alt[:,i] = lla[:,2]

        #lon = lon.round(decimals=5)
        #lat = lat.round(decimals=5)

        # DETTE VIRKER:
        lonmin = np.array([lon0.min(), lon1.min(), lon2.min(), lon3.min(), lon4.min()]).min()
        lonmax = np.array([lon0.max(), lon1.max(), lon2.max(), lon3.max(), lon4.max()]).max()
        latmin = np.array([lat0.min(), lat1.min(), lat2.min(), lat3.min(), lat4.min()]).min()
        latmax = np.array([lat0.max(), lat1.max(), lat2.max(), lat3.max(), lat4.max()]).max()
        if nx is None:
            nx = nn[0].shape()[1] + nn[1].shape()[1] + nn[2].shape()[1] + nn[3].shape()[1] + \
                nn[4].shape()[1]
        # prepare geospatial grid
        merged = Nansat.from_domain(
                Domain(NSR(EPSG), '-lle %f %f %f %f -ts %d %d' % (lonmin, latmin,
                        lonmax, latmax, nx, ny)))

        ## DETTE VIRKER IKKE..
        #merged = Nansat.from_domain(Domain.from_lonlat(lon, lat, add_gcps=False))
        #merged.add_band(array = np.rad2deg(alpha), parameters={'wkv': 'sensor_view'})

        dfdg = np.ones((self.N_SUBSWATHS))*5 # Hz (5 Hz a priori)
        for i in range(self.N_SUBSWATHS):
            dfdg[i] = nn[i].get_uncertainty_of_fdg()
            # TODO: check if 
            nn[i].reproject(merged, tps=True, resample_alg=1, block_size=2)
            #nn[i].reproject(merged, resample_alg=1, block_size=2)
        
        # Initialize band arrays
        inc = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        topo = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        fdg = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        fww = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        ur = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        valid_sea_dop = np.ones(
                (self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        std_fdg = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        std_fww = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        std_ur = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan

        lut_dataset_ids = ''
        for ii in range(self.N_SUBSWATHS):
            inc[ii] = nn[ii]['incidence_angle']
            topo[ii] = nn[ii]['topographic_height']
            fdg[ii] = nn[ii]['fdg']
            fww[ii] = nn[ii]['fww']
            std_fww[ii] = nn[ii]['std_fww']
            ur[ii] = nn[ii]['Ur']
            valid_sea_dop[ii] = nn[ii]['valid_sea_doppler']
            # uncertainty of fdg is a scalar
            std_fdg[ii] = dfdg[ii]*nn[ii]['swathmask']
            # uncertainty of ur
            std_ur[ii] = nn[ii].get_uncertainty_of_radial_current(dfdg[ii])
            # Set lut dataset ids
            lut_dataset_ids += nn[ii].get_metadata('related_dataset_id') + ', '

        merged.set_metadata(key='related_dataset_id', value=lut_dataset_ids[:-2])
        merged.set_metadata(key='related_dataset_relation_type',
                            value='auxiliary,auxiliary,auxiliary,auxiliary,auxiliary')

        # Calculate incidence angle as a simple average
        mean_inc = np.nanmean(inc, axis=0)
        wkv = 'angle_of_incidence'
        merged.add_band(array = mean_inc, parameters={'name': 'incidence_angle',
            'wkv': wkv})

        # Calculate topography as a simple average
        mean_topo = np.nanmean(topo, axis=0)
        wkv = 'height_above_reference_ellipsoid'
        merged.add_band(array = mean_topo, parameters={'name': 'topo',
            'wkv': wkv})

        # Calculate fdg as weighted average
        mean_fdg = nansumwrapper((fdg/np.square(std_fdg)).data, axis=0) / \
                nansumwrapper((1./np.square(std_fdg)).data, axis=0)
        wkv = 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_velocity'
        merged.add_band(array = mean_fdg, parameters={
            'name': 'fdg',
            'wkv': wkv
        })

        # Calculate total surface velocity
        k = 2.*np.pi / ASAR_WAVELENGTH
        merged.add_band(
            array = -np.pi*mean_fdg/(k*np.sin(np.deg2rad(mean_inc))),
            parameters = {
                'name': 'radvel',
                'long_name': 'Total radial surface velocity',
                'units': 'm s-1'
            })

        # Standard deviation of fdg
        std_mean_fdg = np.sqrt(1. / nansumwrapper((1./np.square(std_fdg)).data, axis=0))
        merged.add_band(array = std_mean_fdg, parameters={'name': 'std_fdg', 'units': 'Hz'})

        # Calculate fww as weighted average
        mean_fww = nansumwrapper((fww/np.square(std_fww)).data, axis=0) / \
                nansumwrapper((1./np.square(std_fww)).data, axis=0)
        merged.add_band(array = mean_fww, parameters={
            'name': 'fww',
            'long_name': 'Radar Doppler frequency shift due to wind waves',
            'units': 'Hz',
        })

        # Standard deviation of fww
        std_mean_fww = np.sqrt(1. / nansumwrapper((1./np.square(std_fww)).data, axis=0))
        merged.add_band(array = std_mean_fww, parameters={'name': 'std_fww', 'units': 'Hz'})

        # Calculate ur as weighted average
        mean_ur = nansumwrapper((ur/np.square(std_ur)).data, axis=0) / \
                nansumwrapper((1./np.square(std_ur)).data, axis=0)
        merged.add_band(
            array = mean_ur,
            parameters={
                'name': 'Ur',
                'long_name': 'Surface current velocity',
                'units': 'm s-1'
            }
        )

        # Standard deviation of Ur
        std_mean_ur = np.sqrt(1. / nansumwrapper((1./np.square(std_ur)).data, axis=0))
        merged.add_band(array = std_mean_ur, parameters={'name': 'std_ur', 'units': 'm s-1'})

        # Band of valid pixels
        vsd = np.nanmin(valid_sea_dop, axis=0)
        merged.add_band(
            array = vsd,
            parameters={
                'name': 'valid_sea_doppler',
            }
        )

        # Add file to db
        merged.filename = path_to_nc_file(ds, os.path.basename(nansat_filename(
            ds.dataseturi_set.get(uri__endswith='.gsar').uri)).split('.')[0] + '_merged.nc')
        merged.set_metadata(key='originating_file',
                value=nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri))

        title = (
            'Calibrated geophysical %s %s wide-swath range Doppler frequency '
            'shift retrievals in %s polarisation, %s') %(
                    pti.get_gcmd_platform('envisat')['Short_Name'],
                    pti.get_gcmd_instrument('asar')['Short_Name'],
                    pol,
                    nn[0].get_metadata('time_coverage_start')
            )
        merged.set_metadata(key='title', value=title)
        title_no = (
            'Kalibrert geofysisk %s %s Dopplerskift i full bildebredde og '
            '%s polarisering, %s') %(
                pti.get_gcmd_platform('envisat')['Short_Name'],
                pti.get_gcmd_instrument('asar')['Short_Name'],
                pol,
                nn[0].get_metadata('time_coverage_start')
        )
        merged.set_metadata(key='title_no', value=title_no)

        summary = (
            'Calibrated geophysical %s %s wide-swath range Doppler frequency shift '
            'retrievals in %s polarization. The data was acquired on '
            '%s.') % (
                pti.get_gcmd_platform('envisat')['Short_Name'],
                pti.get_gcmd_instrument('asar')['Short_Name'],
                pol,
                nn[0].get_metadata('time_coverage_start')
            )
        merged.set_metadata(key='summary', value=summary)
        summary_no = (
            'Kalibrert geofysisk %s %s Dopplerskift i full bildebredde og %s '
            'polarisering. Dataene ble samlet %s.') % (
                pti.get_gcmd_platform('envisat')['Short_Name'],
                pti.get_gcmd_instrument('asar')['Short_Name'],
                pol,
                nn[0].get_metadata('time_coverage_start')
            )
        merged.set_metadata(key='summary_no', value=summary_no)
        
        new_uri, created = self.export2netcdf(merged, ds, filename=merged.filename,
                                              history_message=history_message)
        connection.close()

        return merged

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

