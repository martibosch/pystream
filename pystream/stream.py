import numpy as np
import richdem

__all__ = ['StreamSimulation']


class StreamSimulation:
    """
    Classification from Seppelt 2003
    simulation process: precipitation->runoff
    variables: flow
    characteristic time: 1 day or 1 month
    mathematical model: differential equations, translated into discrete
    arithmetic operations on a raster + flow accumulation on a network
    (of pixels)

    Spatial scale:
        - extent: watershed
        - grid size: DEM resolution

    Conceptual model: ABC model ("Streamflow synthesis", Fiering 1967),
    which focuses on the relationships between precipitation,
    evapotranspiration, groundwater storage, streamflow

    Mechanistic model (instead of empirical fits e.g., Machine Learning)
    """

    def __init__(self, dem, slope, whc, cropf, res):
        if isinstance(dem, richdem.rdarray):
            self.dem = dem
        else:
            # TODO: ensure that DEM is an ndarray
            self.dem = richdem.rdarray(dem, no_data=-1)
        # TODO: automatically initialize slope arr from DEM
        self.slope = slope
        # TODO: what to do with WHCcal
        # avoid division by zero
        self.whc = whc
        self.whc[whc <= 0] = .01  # epsilon
        self.cropf = cropf
        # TODO: allow fp as raster inputs and automatically extract resolution
        # from tif
        self.res = res

        # TODO: verbose mode where the state variables are saved for each step
        # start with ones, which justifies why the model needs warmup
        self.snow_accum = np.zeros_like(dem)
        self.available_water = np.zeros_like(dem)
        self.ground_water = np.zeros_like(dem)

        self.MELTcal = 15
        self.CROPFcal = 1.5
        self.HEATcal = .2
        self.TOGWcal = .5
        self.Ccal = 2
        self.SNOWcal = 3
        # TODO: self.WHCcal = 1
        # not to be calibrated, but must be there and the user might change it
        self.TEMP_SNOW = 0

        # TODO: yearly heat, A as class attributes?

        # TODO: self.flux_i
        # TODO: self.time_step

    def simulation_step(self, prec_i, temp_avg_i, temp_max_i, heat, A):
        # Snow
        snowfall_i = np.copy(prec_i)  # [kg]
        # at high temp no snow
        snowfall_i[temp_avg_i > self.TEMP_SNOW] = 0
        # add snow accumulated from previous iterations (months)
        snow_accum_i = self.snow_accum + snowfall_i  # [kg]
        # how much snow would melt at each pixel given its temperature
        # ACHTUNG: use maximum temperature here
        snow_melt_i = self.MELTcal * (temp_max_i + self.SNOWcal)  # [kg]
        # at low temp no snow melts ACHTUNG: use maximum temperature here
        snow_melt_i[(temp_max_i + self.SNOWcal) < self.TEMP_SNOW] = 0
        # no more snow can melt than the snow that there actually is
        snow_melt_i = np.minimum(snow_accum_i, snow_melt_i)
        # this is the actual input of liquid water per pixel
        prec_i = prec_i - snowfall_i + snow_melt_i
        # update snow accumulation for next iteration
        self.snow_accum = snow_accum_i - snow_melt_i

        # Potential evotranspiration (Thornthwaite)
        pe_i = np.full_like(temp_max_i, np.nan)
        high_temp_cond = temp_max_i >= 26.5
        pe_i[high_temp_cond] = -415.85 + (32.24 * temp_max_i[high_temp_cond]) \
                               - (.43 * (temp_max_i[high_temp_cond]**2))
        mid_temp_cond = (temp_max_i > 0) & (temp_max_i < 26.5)
        # TODO: how to handle HEATcal: does heat change at any simulation step?
        # pe_i[mid_temp_cond] = 16 * ((10 * (temp_max_i[mid_temp_cond] /
        # heat[mid_temp_cond])) ** A[mid_temp_cond])
        # pe_i[mid_temp_cond] = 16 * (
        #     (10 * np.power(temp_max_i[mid_temp_cond] / heat[mid_temp_cond],
        #                    A[mid_temp_cond])))
        pe_i[mid_temp_cond] = 16 * ((10 * np.power(
            temp_max_i[mid_temp_cond] / self.HEATcal * heat[mid_temp_cond],
            A[mid_temp_cond])))
        pe_i[temp_max_i <= 0] = 0
        pe_i = pe_i * self.cropf * self.CROPFcal

        # Soil storage (Thornthwaite-Mather)
        prec_eff_i = prec_i - pe_i
        # available water for this iteration; `self.available_water`
        # corresponds to the water available at the end of the previous
        # iteration
        available_water_i = np.copy(self.available_water)
        excess_i = np.zeros_like(prec_eff_i)
        # soil is wetting below capacity
        below_cap = self.available_water + prec_eff_i <= self.whc
        excess_i[below_cap] = 0
        available_water_i[below_cap] = self.available_water[below_cap] + \
            prec_eff_i[below_cap]
        # soil is wetting above capacity
        above_cap = self.available_water + prec_eff_i > self.whc
        excess_i[above_cap] = self.available_water[above_cap] + prec_eff_i[
            above_cap] - self.whc[above_cap]
        available_water_i[above_cap] = self.whc[above_cap]
        # soil is drying
        drying = prec_eff_i <= 0
        excess_i[drying] = 0
        available_water_i[drying] = self.available_water[drying] * np.exp(
            prec_eff_i[drying] / self.whc[drying])
        # update available for the next iteration
        self.available_water = available_water_i

        # Separate direct from delayed runoff
        runoff_i = self.TOGWcal * excess_i
        to_ground_water_i = excess_i - runoff_i

        # Volume of ground water and base flow
        ground_water_i = self.ground_water + to_ground_water_i
        base_flow_i = ground_water_i / (self.slope * self.Ccal)  # [mm]
        # update ground water for the next iteration
        self.ground_water = ground_water_i - base_flow_i

        # Discharge (snow melt + runoff + base flow)
        discharge_i = runoff_i + base_flow_i
        discharge_i = (discharge_i / 1000) * self.res[0] * self.res[1]  # [m^3]

        flux_i = richdem.FlowAccumulation(self.dem, method='D8',
                                          weights=discharge_i)

        return flux_i.max()
