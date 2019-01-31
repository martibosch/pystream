import numpy as np
import rasterio
import richdem
import six

__all__ = ['StreamSimulation']


class StreamSimulation:
    def __init__(self, dem, cropf, whc, res=None, nodata=-9999,
                 whc_epsilon=.01, init_parameters={}):

        #
        # LOAD TERRAIN DATA
        #

        # DEM
        if isinstance(dem, six.string_types):
            with rasterio.open(dem) as dem_src:
                self.dem = richdem.rdarray(
                    dem_src.read(1).astype(np.double), no_data=dem_src.nodata)
                # We assert that all rasters are aligned, so this should be
                # the resolution of all rasters. We will be setting it as
                # class attribute every time we read a raster file to ensure
                # that we get the resolution when some terrain data is
                # provided as ndarray
                self.res = dem_src.res
        elif isinstance(dem, np.ndarray):
            # ensure that the self.dem is a `richdem.rdarray` (which is a
            # subclass of `np.ndarray`)
            if isinstance(dem, richdem.rdarray):
                self.dem = dem
            else:
                # ACHTUNG with the nodata argument, since elevation could
                # perfectly take negative values
                self.dem = richdem.rdarray(
                    dem.astype(np.double), no_data=nodata)

        # CROP FACTOR
        if isinstance(cropf, six.string_types):
            with rasterio.open(cropf) as cropf_src:
                self.cropf = cropf_src.read(1)
                self.res = cropf_src.res  # See comment above `dem_src.res`
        elif isinstance(cropf, np.ndarray):
            self.cropf = cropf

        # WATER HOLDING CAPACITY
        if isinstance(whc, six.string_types):
            with rasterio.open(whc) as whc_src:
                self.whc = whc_src.read(1)
                self.res = whc_src.res  # See comment above `dem_src.res`
        elif isinstance(whc, np.ndarray):
            self.whc = whc

        # whc must be strictly positive, so we must replace all zero/negative
        # pixels for an arbitrarily very small value
        self.whc[self.whc <= 0] = whc_epsilon

        if res is not None:
            # If the resolution is explicitly provided, it takes preference
            # over the resolution extracted from any of the rasters
            self.res = res
        else:
            # Ensure that the resolution was provided in at least one raster
            if not hasattr(self, 'res'):
                raise ValueError(
                    "If passing raster arrays (instead of filepaths), the "
                    "resolution must be provided!")

        if not (self.dem.shape == self.cropf.shape == self.whc.shape):
            raise ValueError("Raster shapes do not match!")

        # TODO: CLIMATOLOGICAL DATA

        # STATE VARIABLES
        # TODO: verbose mode where the state variables are saved for each step
        # start with ones, which justifies why the model needs warmup
        # ACHTUNG: this is the richdem DEM, so it is better not to use
        # `zeros_like` (because it would also return an rdarray). On the other
        # hand, we can enforce a `np.double` data type in order to be
        # consistent with the DEM's data type (remember that richdem enforces
        # using doubles)
        self.snow_accum = np.zeros(self.dem.shape, dtype=np.double)
        self.available_water = np.zeros(self.dem.shape, dtype=np.double)
        self.ground_water = np.zeros(self.dem.shape, dtype=np.double)

        # PARAMETERS
        self.TEMP_SNOW_FALL = init_parameters.get('TEMP_SNOW_FALL', 2)
        self.TEMP_SNOW_MELT = init_parameters.get('TEMP_SNOW_MELT', 0)
        self.SNOW_MELT_COEFF = init_parameters.get('SNOW_MELT_COEFF', 15)
        self.CROPF_COEFF = init_parameters.get('CROPF_COEFF', 1.5)
        self.WHC_COEFF = init_parameters.get('WHC_COEFF', 1.5)
        self.TOGW = init_parameters.get('TOGW', .5)
        self.C = init_parameters.get('C', .2)

        # TODO: self.flux_i
        # TODO: self.time_step

    def simulation_step(self, prec_i, temp_i, year_heat_index, year_alpha,
                        daylight_hours=12):

        # SNOW
        # snow accumulation from the previous iteration
        snow_accum_prev = self.snow_accum
        # Snow
        snowfall_i = np.copy(prec_i)  # [kg]
        # at high temp no snow
        snowfall_i[temp_i > self.TEMP_SNOW_FALL] = 0
        # add snow accumulated from previous iterations (months)
        snow_accum_i = snow_accum_prev + snowfall_i  # [kg]
        # how much snow would melt at each pixel given its temperature
        # ACHTUNG: use maximum temperature here
        snow_melt_i = self.SNOW_MELT_COEFF * (temp_i - self.TEMP_SNOW_MELT
                                              )  # [kg]
        # at low temp no snow melts ACHTUNG: use maximum temperature here
        snow_melt_i[temp_i < self.TEMP_SNOW_MELT] = 0
        # no more snow can melt than the snow that there actually is
        snow_melt_i = np.minimum(snow_accum_i, snow_melt_i)
        # substract the melted snow from the snow accumulation
        snow_accum_i = snow_accum_i - snow_melt_i
        # this is the actual input of liquid water per pixel
        liquid_prec_i = prec_i - snowfall_i + snow_melt_i
        # update snow accumulation for next iteration
        self.snow_accum = snow_accum_i

        # POTENTIAL EVAPOTRANSPIRATION (Thornthwaite)
        pe_i = np.full_like(temp_i, np.nan)
        high_temp_cond = temp_i >= 26.5
        pe_i[high_temp_cond] = -415.85 + 32.24 * temp_i[high_temp_cond] - \
            .43 * temp_i[high_temp_cond]**2
        mid_temp_cond = (temp_i > 0) & (temp_i < 26.5)
        pe_i[mid_temp_cond] = 16 * (
            (10 * (temp_i[mid_temp_cond] / year_heat_index[mid_temp_cond]))
            ** year_alpha[mid_temp_cond])  # yapf: disable
        pe_i[temp_i <= 0] = 0
        pe_i *= (daylight_hours / 12) * self.cropf * self.CROPF_COEFF

        # SOIL STORAGE (Thornthwaite-Mather)
        # here we use two copies of the water available at the end of the last
        # iteration, and update it for the next iteration. We could also
        # update it in a piecewise manner according to whether the soil is
        # wetting above/below capacity or drying, avoiding to create the
        # `avaiable_water_i` variable as in:
        # self.available_water[below_cap] = available_water_prev[below_cap] + \
        #    prec_eff_i[below_cap]
        # self.available_water[above_cap] = whc[above_cap]
        # available_water_i[drying] = available_water_prev[drying] * np.exp(
        #    prec_eff_i[drying] / whc[drying])
        # but I think it is better to update the state variables in one and
        # only one line
        available_water_prev = np.copy(self.available_water)
        available_water_i = np.copy(self.available_water)
        # this is the effective precipitation at each pixel
        prec_eff_i = liquid_prec_i - pe_i
        # water holding capacity at each pixel
        whc = self.whc * self.WHC_COEFF
        # start array of zeros with excess at each pixel
        excess_i = np.zeros_like(prec_eff_i)
        # soil is wetting below capacity
        below_cap = available_water_prev + prec_eff_i <= whc
        excess_i[below_cap] = 0
        available_water_i[below_cap] = available_water_prev[below_cap] + \
            prec_eff_i[below_cap]
        # soil is wetting above capacity
        above_cap = available_water_prev + prec_eff_i > whc
        excess_i[above_cap] = available_water_prev[above_cap] + \
            prec_eff_i[above_cap] - whc[above_cap]
        available_water_i[above_cap] = whc[above_cap]
        # soil is drying
        drying = prec_eff_i <= 0
        excess_i[drying] = 0
        available_water_i[drying] = available_water_prev[drying] * np.exp(
            prec_eff_i[drying] / whc[drying])
        # update available water for next iteration
        self.available_water = available_water_i

        # FLOW SEPARATION
        # separate soil excess that goes to ground water (recharge) and runoff
        runoff_i = (1 - self.TOGW) * excess_i
        to_ground_water_i = excess_i - runoff_i
        ground_water_i = self.ground_water + to_ground_water_i
        # separate ground water that goes to the base flow (discharge)
        base_flow_i = ground_water_i * self.C
        # update ground water for the next iteration
        self.ground_water = ground_water_i - base_flow_i
        # total outflow (snow melt + runoff + base flow) at each pixel (divide
        # by 1000 to convert from liters to m^3)
        outflow_i = runoff_i + base_flow_i
        outflow_i = (outflow_i / 1000) * self.res[0] * self.res[1]

        # FLOW ACCUMULATION
        # weighted flow accumulation to simulate the spatially-explicit stream
        # flow
        streamflow_i = richdem.FlowAccumulation(
            self.dem, method='D8', weights=outflow_i.astype(np.double))

        # Assume that maximum flow corresponds to the gauge station
        gauge_flow_i = streamflow_i.max().item()

        return gauge_flow_i
