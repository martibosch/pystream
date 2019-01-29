import numpy as np
import richdem

__all__ = ['StreamSimulation']


class StreamSimulation:
    def __init__(self, dem, cropf, whc, res, init_parameters={}):
        # TERRAIN DATA
        if isinstance(dem, richdem.rdarray):
            self.dem = dem
        else:
            # TODO: ensure that DEM is an ndarray
            # dem must be double for rdarray
            self.dem = richdem.rdarray(dem.astype(np.double), no_data=-1)
        self.cropf = cropf
        # TODO: what to do with WHCcal
        # avoid division by zero
        self.whc = whc
        self.whc[whc <= 0] = .01  # epsilon
        # TODO: allow fp as raster inputs and automatically extract resolution
        # from tif
        self.res = res

        # TODO: CLIMATOLOGICAL DATA

        # STATE VARIABLES
        # TODO: verbose mode where the state variables are saved for each step
        # start with ones, which justifies why the model needs warmup
        # ACHTUNG: this is the ndarray DEM, not the richdem
        self.snow_accum = np.zeros_like(dem)
        self.available_water = np.zeros_like(dem)
        self.ground_water = np.zeros_like(dem)

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

    def simulation_step(self, prec_i, temp_i, year_heat_index, year_alpha):

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
        pe_i *= self.cropf * self.CROPF_COEFF

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
