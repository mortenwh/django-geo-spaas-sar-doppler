import os, warnings
import json

import numpy as np
from math import sin, pi, cos, acos, copysign
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import median_filter

from dateutil.parser import parse
from datetime import timedelta, datetime
from django.utils import timezone

import netCDF4
from osgeo import ogr, osr
from osgeo.ogr import Geometry

from django.conf import settings
from django.db import models
from django.contrib.gis.geos import WKTReader, Polygon
from django.contrib.gis.geos import Point, MultiPoint
from django.contrib.gis.gdal import OGRGeometry

# Plotting
import matplotlib.pyplot as plt
import cmocean
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# Nansat/geospaas
import pythesint as pti

from geospaas.utils.utils import nansat_filename, media_path, product_path
from geospaas.vocabularies.models import Parameter
from geospaas.catalog.models import GeographicLocation
from geospaas.catalog.models import Dataset, DatasetURI
#from geospaas.viewer.models import Visualization
from geospaas.nansat_ingestor.managers import DatasetManager as DM

from nansat.nansat import Nansat
from nansat.nsr import NSR
from nansat.domain import Domain
from nansat.figure import Figure

from sardoppler.sardoppler import Doppler

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

class DatasetManager(DM):

    N_SUBSWATHS = 5

    def get_or_create(self, uri, *args, **kwargs):
        """ Ingest gsar file to geo-spaas db
        """

        ds, created = super(DatasetManager, self).get_or_create(uri, *args, **kwargs)

        # TODO: Check if the following is necessary
        if not type(ds) == Dataset:
            return ds, False

        fn = nansat_filename(uri)
        n = Nansat(fn, subswath=0)

        # set Dataset entry_title
        ds.entry_title = n.get_metadata('title')
        ds.save()

        if created:
            from sar_doppler.models import SARDopplerExtraMetadata
            # Store the polarization and associate the dataset
            extra, _ = SARDopplerExtraMetadata.objects.get_or_create(dataset=ds,
                    polarization=n.get_metadata('polarization'))
            if not _:
                raise ValueError('Created new dataset but could not create instance of ExtraMetadata')
            ds.sardopplerextrametadata_set.add(extra)

        gg = WKTReader().read(n.get_border_wkt())

        #lon, lat = n.get_border()
        #ind_near_range = 0
        #ind_far_range = int(lon.size/4)
        #import pyproj
        #geod = pyproj.Geod(ellps='WGS84')
        #angle1,angle2,img_width = geod.inv(lon[ind_near_range], lat[ind_near_range], 
        #                                    lon[ind_far_range], lat[ind_far_range])

        # If the area of the dataset geometry is larger than the area of the subswath border, it means that the dataset 
        # has already been created (the area should be the total area of all subswaths)
        if np.floor(ds.geographic_location.geometry.area)>np.round(gg.area):
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
                            ).round(decimals=3)

        poly_border = ','.join(str(llo) + ' ' + str(lla) for llo, lla in zip(lons, lats))
        wkt = 'POLYGON((%s))' % poly_border
        new_geometry = WKTReader().read(wkt)

        # Get or create new geolocation of dataset
        # Returns False if it is the same as an already created one (this may happen when a lot of data is processed)
        ds.geographic_location, cr = GeographicLocation.objects.get_or_create(geometry=new_geometry)

        return ds, True

    def module_name(self):
        """ Get module name
        """
        return self.__module__.split('.')[0]

    def nc_name(self, ds, ii):
        # Filename of exported netcdf
        fn = os.path.join(
                product_path(
                    self.module_name(),
                    nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri)),
                os.path.basename(nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri)).split('.')[0]
                            + 'subswath%s.nc' % ii)
        return fn

    def export2netcdf(self, n, ds, history_message=''):

        if not history_message:
            history_message = 'Export to netCDF [geospaas sar_doppler version %s]' %os.getenv('GEOSPAAS_SAR_DOPPLER_VERSION', 'dev')

        ii = int(n.get_metadata('subswath'))

        date_created = datetime.now(timezone.utc)

        fn = self.nc_name(ds, ii)

        original = Nansat(n.get_metadata('Originating file'), subswath=ii)
        metadata = original.get_metadata()

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
        metadata['Conventions'] = metadata['Conventions'] + ', ACDD-1.3'
        # id - the ID from the database should be registered in the file if it is not already there
        try:
            entry_id = n.get_metadata('entry_id')
        except ValueError:
            n.set_metadata(key='entry_id', value=ds.entry_id)
        try:
            id = n.get_metadata('id')
        except ValueError:
            n.set_metadata(key='id', value=ds.entry_id)
        metadata['date_created'] = date_created.strftime('%Y-%m-%d')
        metadata['date_created_type'] = 'Created'
        metadata['date_metadata_modified'] = date_created.strftime('%Y-%m-%d')
        metadata['processing_level'] = 'Scientific'
        metadata['creator_role'] = 'Investigator'
        metadata['creator_name'] = 'Morten Wergeland Hansen'
        metadata['creator_email'] = 'mortenwh@met.no'
        metadata['creator_institution'] = pretty_print_gcmd_keywords(pti.get_gcmd_provider('NO/MET'))

        metadata['project'] = 'Norwegian Space Agency project JOP.06.20.2: Reprocessing and analysis of historical data for future operationalization of Doppler shifts from SAR'
        metadata['publisher_name'] = 'Morten Wergeland Hansen'
        metadata['publisher_url'] = 'https://www.met.no/'
        metadata['publisher_email'] = 'mortenwh@met.no'

        metadata['references'] = 'https://github.com/mortenwh/openwind'

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

        # history
        try:
            history = n.get_metadata('history')
        except ValueError:
            metadata['history'] = date_created.isoformat() + ': ' + history_message
        else:
            metadata['history'] = history + '\n' + date_created.isoformat() + ': ' + history_message

        # Set metadata from dict (export2thredds could take it as input..)
        for key, val in metadata.items():
            n.set_metadata(key=key, value=val)

        # Export data to netcdf
        print('Exporting %s to %s (subswath %d)' % (n.filename, fn, ii+1))
        n.export(filename=fn)
        #ww.export2thredds(thredds_fn, mask_name='swathmask', metadata=metadata, no_mask_value=1)

        # Clean netcdf attributes
        history = n.get_metadata('history')
        self.clean_nc_attrs(fn, history)

        # Add netcdf uri to DatasetURIs
        ncuri = 'file://localhost' + fn
        new_uri, created = DatasetURI.objects.get_or_create(uri=ncuri, dataset=ds)

    @staticmethod
    def clean_nc_attrs(fn, history):
        ncdataset = netCDF4.Dataset(fn, 'a')

        # Fix issue with gdal overwriting the history
        hh = ncdataset.history
        ncdataset.history = history + '\n' + hh
        ncdataset.delncattr('GDAL_history')

        # Remove obsolete Conventions attribute
        rm_attrs = ['Conventions', 'GDAL_GDAL']
        for rm_attr in rm_attrs:
            if rm_attr in ncdataset.ncattrs():
                ncdataset.delncattr(rm_attr)

        # Remove GDAL_ from attribute names
        strip_str = 'GDAL_'
        for attr in ncdataset.ncattrs():
            if attr.startswith(strip_str) and not 'NANSAT' in attr:
                try:
                    ncdataset.renameAttribute(attr, attr.replace(strip_str,''))
                except:
                    print(attr, attr.replace(strip_str,''))
                    raise
        ncdataset.close()

    def process(self, ds, force=False, *args, **kwargs):
        """ Create data products
        """
        swath_data = {}

        # Set media path (where images will be stored)
        mp = media_path(self.module_name(), nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri))
        # Set product path (where netcdf products will be stored)
        ppath = product_path(
            self.module_name(),
            nansat_filename(ds.dataseturi_set.get(uri__endswith = '.gsar').uri)
        )

        # Loop subswaths, process each of them
        processed = False

        print('Processing %s'%ds)
        # Read subswaths 
        dss = {1: None, 2: None, 3: None, 4: None, 5: None}
        for i in range(self.N_SUBSWATHS):
            processed = True
            # Check if the data has already been processed
            try:
                fn = nansat_filename(ds.dataseturi_set.get(uri__endswith='%d.nc'%i).uri)
            except DatasetURI.DoesNotExist:
                processed = False
            else:
                dd = Nansat(fn)
                try:
                    std_Ur = dd['std_Ur']
                except ValueError:
                    processed = False
            if processed and not force:
                continue
            # Process from scratch to avoid duplication of bands
            fn = nansat_filename(ds.dataseturi_set.get(uri__endswith='.gsar').uri)
            dd = Doppler(fn, subswath=i)

            # Check if the file is corrupted
            try:
                inc = dd['incidence_angle']
            #  TODO: What kind of exception ?
            except:
                processed = False
                continue

            dss[i+1] = dd

        if processed and not force:
            return ds, False

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

	# Get range bias corrected Doppler
        fdg = {}
        fdg[1] = dss[1].anomaly() - dss[1].range_bias()
        fdg[2] = dss[2].anomaly() - dss[2].range_bias()
        fdg[3] = dss[3].anomaly() - dss[3].range_bias()
        fdg[4] = dss[4].anomaly() - dss[4].range_bias()
        fdg[5] = dss[5].anomaly() - dss[5].range_bias()

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

        # Correct by land or mean fww
        wind_fn = nansat_filename(
            Dataset.objects.get(
                source__platform__short_name = 'ERA15DAS',
                time_coverage_start__lte = ds.time_coverage_end,
                time_coverage_end__gte = ds.time_coverage_start
            ).dataseturi_set.get().uri
        )
        land = np.array([])
        fww = np.array([])
        offset_corrected = 0
        for key in dss.keys():
            land = np.append(land, fdg[key][dss[key]['valid_land_doppler'] == 1].flatten())
        if land.any():
            print('Using land for bias corrections')
            land_bias = np.nanmedian(land)
            offset_corrected = 1
        else:
            print('Using CDOP wind-waves Doppler for bias corrections')
            # correct by mean wind doppler
            for key in dss.keys():
                ff = fdg[key].copy()
                # do CDOP correction
                ff[ dss[key]['valid_sea_doppler']==1 ] = \
                    ff[ dss[key]['valid_sea_doppler']==1 ] \
                    - dss[key].wind_waves_doppler(wind_fn)[0][ dss[key]['valid_sea_doppler']==1 ]
                ff[dss[key]['valid_doppler']==0] = np.nan
                fww = np.append(fww, ff.flatten())
            land_bias = np.nanmedian(fww)
            if np.isnan(land_bias):
                offset_corrected = 0
                raise Exception('land bias is NaN...')
            else:
                offset_corrected = 1

        for key in dss.keys():
            fdg[key] -= land_bias
            # Set unrealistically high/low values to NaN (ref issue #4 and #5)
            fdg[key][fdg[key]<-100] = np.nan
            fdg[key][fdg[key]>100] = np.nan
            # Add fdg[key] as band
            dss[key].add_band(
                array=fdg[key],
                parameters={
                    'wkv': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_velocity',
                    'offset_corrected': str(offset_corrected)
                }
            )

            # Add Doppler anomaly
            dss[key].add_band(
                array = dss[key].anomaly(), 
                parameters = {
                    'wkv': 'anomaly_of_surface_backwards_doppler_centroid_frequency_shift_of_radar_wave'
                }
            )

            # Add wind doppler and its uncertainty as bands
            fww, dfww = dss[key].wind_waves_doppler(wind_fn)
            dss[key].add_band(
                array = fww,
                parameters = {
                    'wkv': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_wind_waves'
                }
            )
            dss[key].add_band(
                array = dfww,
                parameters = {'name': 'std_fww'}
            )

            # Calculate range current velocity component
            v_current, std_v, offset_corrected = \
                dss[key].surface_radial_doppler_sea_water_velocity(wind_fn, fdg=fdg[key])
            dss[key].add_band(
                array = v_current,
                parameters = {
                    'wkv': 'surface_radial_doppler_sea_water_velocity',
                    'offset_corrected': str(offset_corrected)
                }
            )
            dss[key].add_band(array=std_v, parameters={'name': 'std_Ur'})
  
            # Set satellite pass
            lon,lat = dss[key].get_geolocation_grids()
            gg = np.gradient(lat, axis=0)
            # DEFS: ascending pass is 1, descending pass is 0
            sat_pass = np.ones(gg.shape) # for ascending pass
            sat_pass[gg<0] = 0
            dss[key].add_band(
                array = sat_pass,
                parameters = {
                    'name': 'sat_pass',
                    'comment': 'ascending pass is 1, descending pass is 0'
                }
            )

            history_message = (
                'sar_doppler.models.Dataset.objects.process("%s") '
                '[geospaas sar_doppler version %s]' % (
                    ds, os.getenv('GEOSPAAS_SAR_DOPPLER_VERSION', 'dev')
                )
            )
            self.export2netcdf(dss[key], ds, history_message=history_message)
            processed = True

        return ds, processed

    def get_merged_swaths(self, ds, resolution=None, EPSG = 4326, lowres=False, **kwargs):
        """Merge swaths and return a Nansat object.

        EPSG options:
            - 4326: WGS 84 / longlat
            - 3995: WGS 84 / Arctic Polar Stereographic
        """
        # Make sure dataset is processed
        ds, cr = self.process(ds, **kwargs)

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
        lonmin = np.array([lon0.min(), lon1.min(), lon2.min(), lon3.min(), lon4.min()]).min()
        lonmax = np.array([lon0.max(), lon1.max(), lon2.max(), lon3.max(), lon4.max()]).max()
        latmin = np.array([lat0.min(), lat1.min(), lat2.min(), lat3.min(), lat4.min()]).min()
        latmax = np.array([lat0.max(), lat1.max(), lat2.max(), lat3.max(), lat4.max()]).max()
        tsx = nn[0].shape()[1] + nn[1].shape()[1] + nn[2].shape()[1] + nn[3].shape()[1] + \
            nn[4].shape()[1]
        tsy = np.array([
            nn[0].shape()[0],
            nn[1].shape()[0],
            nn[2].shape()[0],
            nn[3].shape()[0],
            nn[4].shape()[0]
        ]).max()

        if resolution:
            # TODO: change resolution according to input...
            tsx = tsx
            tsy = tsy

        # prepare geospatial grid
        d = Domain(NSR(EPSG), '-lle %f %f %f %f -ts %d %d' % (lonmin-0.5, latmin-.5,
                        lonmax+.5, latmax+.5, tsx, tsy))
        
        for i in range(self.N_SUBSWATHS):
            nn[i].reproject(d, tps=False)
        
        merged = Nansat.from_domain(nn[0])
        fdg = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        ur = np.ones((self.N_SUBSWATHS, merged.shape()[0], merged.shape()[1])) * np.nan
        fsize = [11, 13, 15, 17, 19]
        for ii in range(self.N_SUBSWATHS):
            if lowres:
                # See filter_size in residual_error.py
                # OBS - this ends up with all nan..
                fdg[ii] = uniform_filter(np.where(np.isnan(nn[ii]['fdg']), -np.inf, nn[ii]['fdg']), size=fsize[ii])
                ur[ii] = uniform_filter(nn[ii]['Ur'], size=fsize[ii])
            else:
                fdg[ii] = nn[ii]['fdg']
                ur[ii] = nn[ii]['Ur']

        fdg = np.nanmean(fdg, axis=0)
        merged.add_band(
            array = fdg,
            parameters={
                'name': 'fdg',
                'wkv': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_velocity'
            }
        )
        ur = np.nanmean(ur, axis=0)
        merged.add_band(
            array = ur,
            parameters={
                'name': 'Ur',
            }
        )
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

