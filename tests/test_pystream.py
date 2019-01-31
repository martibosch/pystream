import unittest

import rasterio
import richdem

import pystream as pst


class TestStreamSimulation(unittest.TestCase):
    def setUp(self):
        self.dem_fp = 'tests/input_data/dem.tif'
        self.cropf_fp = 'tests/input_data/cropf.tif'
        self.whc_fp = 'tests/input_data/whc.tif'
        self.res = (.05, .05)

    def test_io(self):
        with rasterio.open(self.dem_fp) as dem_src:
            dem_arr = dem_src.read(1)
        with rasterio.open(self.cropf_fp) as cropf_src:
            cropf_arr = cropf_src.read(1)
        with rasterio.open(self.whc_fp) as whc_src:
            whc_arr = whc_src.read(1)

        # instantiating a simulation with only raster arrays should raise
        # ValueError because of the missing information on resolution
        self.assertRaises(ValueError, pst.StreamSimulation, dem_arr, cropf_arr,
                          whc_arr)
        # Check that the instance always gets the right resolution
        # if we provide arrays and an explicit resolution, we should be fine
        self.assertEqual(
            pst.StreamSimulation(dem_arr, cropf_arr, whc_arr,
                                 res=self.res).res, self.res)
        # if we explicitly provide a different resolution, it takes preference
        self.assertEqual(
            pst.StreamSimulation(dem_arr, cropf_arr, whc_arr, res=(1, 1)).res,
            (1, 1))
        # we should also be fine if at least one of the raster datasets is
        # provided as filepath, since we will extract the resolution from there
        self.assertEqual(
            pst.StreamSimulation(self.dem_fp, cropf_arr, whc_arr).res,
            self.res)
        self.assertEqual(
            pst.StreamSimulation(dem_arr, self.cropf_fp, whc_arr).res,
            self.res)
        self.assertEqual(
            pst.StreamSimulation(dem_arr, cropf_arr, self.whc_fp).res,
            self.res)

        # instantiating a simulation with rasters of mistmatching shapes
        # should raise a ValueError
        self.assertRaises(ValueError, pst.StreamSimulation, dem_arr[1:, 1:],
                          self.cropf_fp, self.whc_fp)
        self.assertRaises(ValueError, pst.StreamSimulation, self.dem_fp,
                          cropf_arr[1:, 1:], self.whc_fp)
        self.assertRaises(ValueError, pst.StreamSimulation, self.dem_fp,
                          self.cropf_fp, whc_arr[1:, 1:])

        # test that in any case, the dem is an instance of richdem.rdarray
        sts_from_arr = pst.StreamSimulation(dem_arr, cropf_arr, whc_arr,
                                            res=self.res)
        sts_from_fp = pst.StreamSimulation(self.dem_fp, cropf_arr, whc_arr)
        self.assertIsInstance(sts_from_arr.dem, richdem.rdarray)
        self.assertIsInstance(sts_from_fp.dem, richdem.rdarray)

        # finally, ensure that the 'state variables' (i.e., the instance
        # attributes that keep track of the snow accumulated, available water,
        # and ground water at each simulation step) match the shape of the
        # rest of arrays
        dem_from_arr_shape = sts_from_arr.dem.shape
        dem_from_fp_shape = sts_from_fp.dem.shape
        for state_variable in [
                'snow_accum', 'available_water', 'ground_water'
        ]:
            self.assertEqual(
                getattr(sts_from_arr, state_variable).shape,
                dem_from_arr_shape)

            self.assertEqual(
                getattr(sts_from_fp, state_variable).shape, dem_from_fp_shape)
