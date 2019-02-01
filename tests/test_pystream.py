import unittest

import rasterio
import richdem
import xarray as xr

import pystream as pst


class TestMonthlySimulation(unittest.TestCase):
    def setUp(self):
        self.dem_fp = 'tests/input_data/dem.tif'
        self.cropf_fp = 'tests/input_data/cropf.tif'
        self.whc_fp = 'tests/input_data/whc.tif'
        self.prec_fp = 'tests/input_data/prec.nc'
        self.temp_fp = 'tests/input_data/temp.nc'
        self.res = (.05, .05)
        self.num_months = 24

    def test_io_terrain(self):
        with rasterio.open(self.dem_fp) as dem_src:
            dem_arr = dem_src.read(1)
        with rasterio.open(self.cropf_fp) as cropf_src:
            cropf_arr = cropf_src.read(1)
        with rasterio.open(self.whc_fp) as whc_src:
            whc_arr = whc_src.read(1)

        # instantiating a simulation with only raster arrays should raise
        # ValueError because of the missing information on resolution
        self.assertRaises(ValueError, pst.MonthlySimulation, dem_arr,
                          cropf_arr, whc_arr, self.prec_fp, self.temp_fp)
        # Check that the instance always gets the right resolution
        # if we provide arrays and an explicit resolution, we should be fine
        self.assertEqual(
            pst.MonthlySimulation(dem_arr, cropf_arr, whc_arr, self.prec_fp,
                                  self.temp_fp, res=self.res).res, self.res)
        # if we explicitly provide a different resolution, it takes preference
        self.assertEqual(
            pst.MonthlySimulation(dem_arr, cropf_arr, whc_arr, self.prec_fp,
                                  self.temp_fp, res=(1, 1)).res, (1, 1))
        # we should also be fine if at least one of the raster datasets is
        # provided as filepath, since we will extract the resolution from there
        self.assertEqual(
            pst.MonthlySimulation(self.dem_fp, cropf_arr, whc_arr,
                                  self.prec_fp, self.temp_fp).res, self.res)
        self.assertEqual(
            pst.MonthlySimulation(dem_arr, self.cropf_fp, whc_arr,
                                  self.prec_fp, self.temp_fp).res, self.res)
        self.assertEqual(
            pst.MonthlySimulation(
                dem_arr,
                cropf_arr,
                self.whc_fp,
                self.prec_fp,
                self.temp_fp,
            ).res, self.res)

        # instantiating a simulation with rasters of mistmatching shapes
        # should raise a ValueError
        self.assertRaises(ValueError, pst.MonthlySimulation, dem_arr[1:, 1:],
                          self.cropf_fp, self.whc_fp, self.prec_fp,
                          self.temp_fp)
        self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
                          cropf_arr[1:, 1:], self.whc_fp, self.prec_fp,
                          self.temp_fp)
        self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
                          self.cropf_fp, whc_arr[1:, 1:], self.prec_fp,
                          self.temp_fp)

        # test that in any case, the dem is an instance of richdem.rdarray
        ms_from_arr = pst.MonthlySimulation(dem_arr, cropf_arr, whc_arr,
                                            self.prec_fp, self.temp_fp,
                                            res=self.res)
        ms_from_fp = pst.MonthlySimulation(self.dem_fp, cropf_arr, whc_arr,
                                           self.prec_fp, self.temp_fp)
        self.assertIsInstance(ms_from_arr.dem, richdem.rdarray)
        self.assertIsInstance(ms_from_fp.dem, richdem.rdarray)

        # ensure that the 'state variables' (i.e., the instance attributes
        # that keep track of the snow accumulated, available water, and ground
        # water at each simulation step) match the shape of the rest of arrays
        dem_from_arr_shape = ms_from_arr.dem.shape
        dem_from_fp_shape = ms_from_fp.dem.shape
        for state_variable in [
                'snow_accum', 'available_water', 'ground_water'
        ]:
            self.assertEqual(
                getattr(ms_from_arr, state_variable).shape, dem_from_arr_shape)

            self.assertEqual(
                getattr(ms_from_fp, state_variable).shape, dem_from_fp_shape)

    def test_io_climatological(self):
        # test that the varnames and number of months are extracted correctly
        # from filepaths
        ms = pst.MonthlySimulation(self.dem_fp, self.cropf_fp, self.whc_fp,
                                   self.prec_fp, self.temp_fp)
        self.assertEqual(ms.prec_varname, 'prec')
        self.assertEqual(ms.temp_varname, 'temp')
        self.assertEqual(ms.num_months, self.num_months)

        # test that passing wrong varnames raises a ValueError
        self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
                          self.cropf_fp, self.whc_fp, self.prec_fp,
                          self.temp_fp, prec_varname='foo')
        self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
                          self.cropf_fp, self.whc_fp, self.prec_fp,
                          self.temp_fp, temp_varname='foo')

        # now test messing with the datasets
        prec_ds = xr.open_dataset(self.prec_fp, decode_times=False)
        temp_ds = xr.open_dataset(self.temp_fp, decode_times=False)

        # TODO: test that there must be a time dimension and (x, y)/(lon, lat)
        # coordinates, and that they match the terrain data
        # self.assertRaises(ValueError, pst.MonthlySimulation,
        #                   self.dem_fp, self.cropf_fp, self.whc_fp,
        #                   prec_ds.drop('time'), temp_ds)
        # self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
        #                   self.cropf_fp, self.whc_fp, prec_ds,
        #                   temp_ds.drop('time'))

        # test that time dimensions match
        wrong_slice = slice(0, self.num_months - 1)
        self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
                          self.cropf_fp, self.whc_fp,
                          prec_ds.isel(time=wrong_slice), temp_ds)
        self.assertRaises(ValueError, pst.MonthlySimulation, self.dem_fp,
                          self.cropf_fp, self.whc_fp, prec_ds,
                          temp_ds.isel(time=wrong_slice))
