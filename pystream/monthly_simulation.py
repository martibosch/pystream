import itertools

import numpy as np
import rasterio
import richdem
import xarray as xr

from . import plotting, utils

__all__ = ['MonthlySimulation']


class MonthlySimulation:
    # TODO: more flexible approach
    TIME_STEP = 2592000  # i.e., 30 * 24 * 3600 seconds per month

    @staticmethod
    def _prepare_ds(filepath_or_dataset, varname, decode_times):
        if isinstance(filepath_or_dataset, str):
            ds = xr.open_dataset(filepath_or_dataset,
                                 decode_times=decode_times)
        elif isinstance(filepath_or_dataset, xr.Dataset):
            ds = filepath_or_dataset
        # TODO: raise ValueError if `filepath_or_dataset` is not str nor
        # xr.Dataset?

        data_vars = ds.data_vars
        if varname is None:
            # get the name of the first variable, assert that the user has
            # provided an appropriate dataset
            for varname, _ in data_vars.items():
                break
        else:
            if varname not in data_vars:
                raise ValueError(
                    f"Variable {varname} must be among {data_vars}")

        return ds, varname

    def __init__(self, dem, cropf, whc, prec, temp,
                 monthly_daylight_hours=None, prec_varname=None,
                 temp_varname=None, res=None, nodata=-9999, whc_epsilon=.01,
                 decode_times=False, init_parameters={}):

        #
        # LOAD TERRAIN DATA
        #

        # DEM
        if isinstance(dem, str):
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
        if isinstance(cropf, str):
            with rasterio.open(cropf) as cropf_src:
                self.cropf = cropf_src.read(1)
                self.res = cropf_src.res  # See comment above `dem_src.res`
        elif isinstance(cropf, np.ndarray):
            self.cropf = cropf

        # WATER HOLDING CAPACITY
        if isinstance(whc, str):
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

        # TODO: global shape class attribute raising ValueError to ensure
        # consistent shapes?
        if not (self.dem.shape == self.cropf.shape == self.whc.shape):
            raise ValueError("Raster shapes do not match!")

        #
        # CLIMATOLOGICAL DATA
        #

        # TODO: support ndarrays as climatological data?

        # PRECIPITATION
        self.prec_ds, self.prec_varname = MonthlySimulation._prepare_ds(
            prec, prec_varname, decode_times)

        # TEMPERATURE
        self.temp_ds, self.temp_varname = MonthlySimulation._prepare_ds(
            temp, temp_varname, decode_times)

        # we assert that not only the `time` dimensions match, but so do the
        # (x, y)/(lon, lat) coordinates. TODO: enforce it by raising
        # ValueError otherwise? Maybe add a class attribute with the shape
        # (tuple) of TERRAIN and CLIMATOLOGICAL data
        if len(self.prec_ds['time']) == len(self.temp_ds['time']):
            self.num_months = len(self.prec_ds['time'])
        else:
            raise ValueError(
                "Time dimensions of climatological datasets do not match")

        # OTHER

        # this will be used later in the `simulate` method
        if monthly_daylight_hours:
            self.monthly_daylight_hours = monthly_daylight_hours

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
        self.HEAT_COEFF = init_parameters.get('HEAT_COEFF', 1)
        self.TEMP_SNOW_FALL = init_parameters.get('TEMP_SNOW_FALL', 2)
        self.TEMP_SNOW_MELT = init_parameters.get('TEMP_SNOW_MELT', 0)
        self.SNOW_MELT_COEFF = init_parameters.get('SNOW_MELT_COEFF', 15)
        self.CROPF_COEFF = init_parameters.get('CROPF_COEFF', 1.5)
        self.WHC_COEFF = init_parameters.get('WHC_COEFF', 1.5)
        self.TOGW = init_parameters.get('TOGW', .5)
        self.C = init_parameters.get('C', .2)

        # TODO: self.flux_i
        # TODO: self.time_step

    @staticmethod
    def _compute_alpha(heat_index):
        # Thornthwaite (1948)
        return .49239 + .01792 * heat_index - .0000771771 * heat_index**2 \
            + .000000675 * heat_index**3

    # this is the STREAM model's core
    def _simulation_step(self, prec_i, temp_i, year_heat_index, year_alpha,
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

    def simulate(self, heat_index=None, alpha=None):

        # iterator that yields the monthly daylight hours
        try:
            daylight_hours_pool = itertools.cycle(self.monthly_daylight_hours)
        except AttributeError:
            # if the monthly daylight hours were not provided, we assert that
            # in every month, every day has 12 hours of light
            daylight_hours_pool = itertools.cycle([12])

        gauge_flow = np.zeros(self.num_months)

        if heat_index is not None:
            if alpha is None:
                alpha = MonthlySimulation._compute_alpha(heat_index)

            for i in range(self.num_months):
                gauge_flow[i] = self._simulation_step(
                    self.prec_ds.isel(time=i)[self.prec_varname].values,
                    self.temp_ds.isel(time=i)[self.temp_varname].values,
                    heat_index, alpha, next(daylight_hours_pool))
        else:
            # Calculate yearly heat index and alpha using Thornthwaite's
            # equation
            num_years = self.num_months // 12
            if self.num_months % 12 != 0:
                raise ValueError(
                    "The heat index can only be computed for an entire year! "
                    "Ensure that your climatological datasets start have a "
                    "number of months that is multiple of 12")

            for year in range(num_years):
                year_first_month = year * 12
                year_last_month = year_first_month + 12
                year_temp_ds = self.temp_ds.isel(
                    time=slice(year_first_month, year_last_month))

                year_heat_index = year_temp_ds[self.temp_varname] \
                    .groupby('time').apply(lambda temp: (temp / 5) ** 1.514) \
                    .fillna(0).sum('time').values
                # in this case, use the coefficient
                year_heat_index *= self.HEAT_COEFF

                year_alpha = MonthlySimulation._compute_alpha(year_heat_index)

                for i in range(year_first_month, year_last_month):
                    gauge_flow[i] = self._simulation_step(
                        self.prec_ds.isel(time=i)[self.prec_varname].values,
                        self.temp_ds.isel(time=i)[self.temp_varname].values,
                        year_heat_index, year_alpha, next(daylight_hours_pool))

        # from m^3 to m^3/s
        gauge_flow /= self.TIME_STEP

        # set it as class attribute in case they want to plot it later
        self.gauge_flow = gauge_flow

        return gauge_flow

    def plot_gauge_flow(self, obs_gauge_flow=None, num_warmup_months=6,
                        **kwargs):
        return plotting.plot_gauge_flow(
            self.gauge_flow, obs_gauge_flow=obs_gauge_flow,
            num_warmup_months=num_warmup_months, **kwargs)

    def nash_sutcliffe(self, obs_gauge_flow, num_warmup_months=6):
        return utils.nash_sutcliffe(self.gauge_flow[num_warmup_months:],
                                    obs_gauge_flow[num_warmup_months:])
