import os, warnings
from math import sin, pi, cos, acos, copysign
import numpy as np
from scipy.ndimage.filters import median_filter

from dateutil.parser import parse
from datetime import timedelta

from osgeo.ogr import Geometry

from django.conf import settings
from django.utils import timezone
from django.db import models
from django.contrib.gis.geos import WKTReader, Polygon
from django.contrib.gis.geos import Point, MultiPoint
from django.contrib.gis.gdal import OGRGeometry

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

class DatasetManager(DM):

    N_SUBSWATHS = 5

    def get_or_create(self, uri, *args, **kwargs):
        """ Ingest gsar file to geo-spaas db
        """

        ds, created = super(DatasetManager, self).get_or_create(uri, *args, **kwargs)

        # TODO: Check if the following is necessary
        if not type(ds) == Dataset:
            return ds, False

        # set Dataset entry_title
        ds.entry_title = 'SAR Doppler'
        ds.save()

        fn = nansat_filename(uri)
        n = Nansat(fn, subswath=0)
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
                            ).round(decimals=4)

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
                            ).round(decimals=4)

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

    def export2netcdf(self, n, ds):

        i = n.get_metadata('subswath')

        # Set filename of exported netcdf
        fn = os.path.join(product_path(self.module_name(), n.filename),
                            os.path.basename(n.filename).split('.')[0]
                            + 'subswath%s.nc' % i)
        # Set filename of original gsar file in metadata
        n.set_metadata(key='Originating file',
                                        value=n.filename)
        # Export data to netcdf
        print('Exporting %s (subswath %s)' % (n.filename, i))
        n.export(filename=fn)

        # Add netcdf uri to DatasetURIs
        ncuri = 'file://localhost' + fn
        #sjekk ncuri og ds
        new_uri, created = DatasetURI.objects.get_or_create(uri=ncuri,
                                                            dataset=ds)

    def process(self, uri, *args, **kwargs):
        """ Create data products
        """
        ds, created = self.get_or_create(uri, *args, **kwargs)
        fn = nansat_filename(uri)
        swath_data = {}
        # Read subswaths 
        for i in range(self.N_SUBSWATHS):
            swath_data[i] = Doppler(fn, subswath=i)

        # Set media path (where images will be stored)
        mp = media_path(self.module_name(), swath_data[i].filename)
        # Set product path (where netcdf products will be stored)
        ppath = product_path(self.module_name(), swath_data[i].filename)

        # Loop subswaths, process each of them and create figures for display with leaflet
        processed = True
        for i in range(self.N_SUBSWATHS):
            # Check if the file is corrupted
            try:
                inci = swath_data[i]['incidence_angle']
            #  TODO: What kind of exception ?
            except:
                processed = False
                continue

            # Add Doppler anomaly
            swath_data[i].add_band(array=swath_data[i].anomaly(), parameters={
                'wkv':
                'anomaly_of_surface_backwards_doppler_centroid_frequency_shift_of_radar_wave'
            })

            # Get band number of DC freq, then DC polarisation
            band_number = swath_data[i].get_band_number({
                'standard_name': 'surface_backwards_doppler_centroid_frequency_shift_of_radar_wave',
                })
            pol = swath_data[i].get_metadata(band_id=band_number, key='polarization')

            # Calculate total geophysical Doppler shift
            fdg = swath_data[i].geophysical_doppler_shift()
            swath_data[i].add_band(
                array=fdg,
                parameters={
                    'wkv': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_velocity'
                })

            self.export2netcdf(swath_data[i], ds)

            # Reproject to leaflet projection
            xlon, xlat = swath_data[i].get_corners()
            d = Domain(NSR(3857),
                       '-lle %f %f %f %f -tr 1000 1000'
                       % (xlon.min(), xlat.min(), xlon.max(), xlat.max()))
            swath_data[i].reproject(d, resample_alg=1, tps=True)

            # Check if the reprojection failed
            try:
                inci = swath_data[i]['incidence_angle']
            except:
                processed = False
                warnings.warn('Could not read incidence angles - reprojection failed')
                continue

            # Create visualizations of the following bands (short_names)
            ingest_creates = ['valid_doppler',
                              'valid_land_doppler',
                              'valid_sea_doppler',
                              'dca',
                              'fdg']
            for band in ingest_creates:
                filename = '%s_subswath_%d.png' % (band, i)
                # check uniqueness of parameter
                param = Parameter.objects.get(short_name=band)
                if swath_data[i].filename == \
                        '/mnt/10.11.12.232/sat_downloads_asar/level-0/2010-01/ascending/HH/gsar_rvl/RVL_ASA_WS_20100119213139150.gsar' \
                    or swath_data[i].filename == \
                        '/mnt/10.11.12.232/sat_downloads_asar/level-0/2010-01/descending/HH/gsar_rvl/RVL_ASA_WS_20100129225931627.gsar':
                    # generates memory error in write_figure...
                    continue
                fig = swath_data[i].write_figure(
                    os.path.join(mp, filename),
                    bands=band,
                    mask_array=swath_data[i]['swathmask'],
                    mask_lut={0: [128, 128, 128]},
                    transparency=[128, 128, 128])

                if type(fig) == Figure:
                    print('Created figure of subswath %d, band %s' % (i, band))
                else:
                    warnings.warn('Figure NOT CREATED')

                ## Create GeographicLocation for the visualization object
                #geom, created = GeographicLocation.objects.get_or_create(
                #        geometry=WKTReader().read(swath_data[i].get_border_wkt()))

                ## Create Visualization
                #vv, created = Visualization.objects.get_or_create(
                #    uri='file://localhost%s/%s' % (mp, filename),
                #    title='%s (swath %d)' % (param.standard_name, i + 1),
                #    geographic_location=geom
                #)

        # TODO: consider merged figures like Jeong-Won has added in the development branch

        return ds, processed

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

