import numpy as np
import richdem

from . import models

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
        # ACHTUNG: this is the ndarray DEM, not the richdem
        self.snow_accum = np.zeros_like(dem)
        self.available_water = np.zeros_like(dem)
        self.ground_water = np.zeros_like(dem)

        # Parameters
        self.TEMP_SNOW_MELT = 0
        self.SNOW_MELT_COEFF = 15
        self.SNOW_MELT_SHIFT = 3

        self.CROPF_COEFF = 1.5

        self.HEATcal = .2
        self.TOGWcal = .5
        self.Ccal = 2
        # TODO: self.WHCcal = 1
        # not to be calibrated, but must be there and the user might change it

        # TODO: yearly heat, A as class attributes?

        # TODO: self.flux_i
        # TODO: self.time_step

    def simulation_step(self, prec_i, temp_avg_i, temp_max_i, heat, A):

        # Snow
        # snow accumulation from the previous iteration
        snow_accum_prev = self.snow_accum
        liquid_prec_i, snow_accum_i = models.snow(
            prec_i, temp_avg_i, temp_max_i, snow_accum_prev,
            self.TEMP_SNOW_MELT, self.SNOW_MELT_COEFF, self.SNOW_MELT_SHIFT)
        # update snow accumulation for next iteration
        self.snow_accum = snow_accum_i
        # return liquid_prec_i, snow_accum_i

        # Potential evotranspiration (Thornthwaite)
        pe_i = models.potential_evapotranspiration(
            temp_max_i, heat, A, self.cropf, self.CROPF_COEFF)
        # return pe_i

        # Soil storage (Thornthwaite-Mather)
        excess_i, available_water_i = models.soil_storage(
            liquid_prec_i, pe_i, self.available_water, self.whc)
        # update available water for next iteration
        self.available_water = available_water_i
        # return excess_i, available_water_i

        # TODO: put this in models
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

        flow_i = richdem.FlowAccumulation(self.dem, method='D8',
                                          weights=discharge_i)

        # Assume that maximum flow corresponds to the gauge station
        gauge_flow_i = flow_i.max().item()

        return gauge_flow_i
