import numpy as np


def snow(prec_i, temp_avg_i, temp_max_i, snow_accum_prev, TEMP_SNOW_MELT,
         SNOW_MELT_COEFF, SNOW_MELT_SHIFT):
    # TODO: separate TEMP_SNOW_FALL from TEMP_SNOW_MELT?
    # Snow
    snowfall_i = np.copy(prec_i)  # [kg]
    # at high temp no snow
    snowfall_i[temp_avg_i > TEMP_SNOW_MELT] = 0
    # add snow accumulated from previous iterations (months)
    snow_accum_i = snow_accum_prev + snowfall_i  # [kg]
    # how much snow would melt at each pixel given its temperature
    # ACHTUNG: use maximum temperature here
    snow_melt_i = SNOW_MELT_COEFF * (temp_max_i + SNOW_MELT_SHIFT)  # [kg]
    # at low temp no snow melts ACHTUNG: use maximum temperature here
    snow_melt_i[(temp_max_i + SNOW_MELT_SHIFT) < TEMP_SNOW_MELT] = 0
    # no more snow can melt than the snow that there actually is
    snow_melt_i = np.minimum(snow_accum_i, snow_melt_i)
    # substract the melted snow from the snow accumulation
    snow_accum_i = snow_accum_i - snow_melt_i
    # this is the actual input of liquid water per pixel
    liquid_prec_i = prec_i - snowfall_i + snow_melt_i

    return liquid_prec_i, snow_accum_i


def potential_evapotranspiration(temp_max_i, heat, A, cropf, CROPF_COEFF):
    pe_i = np.full_like(temp_max_i, np.nan)
    high_temp_cond = temp_max_i >= 26.5
    pe_i[high_temp_cond] = -415.85 + 32.24 * temp_max_i[high_temp_cond] - \
        .43 * temp_max_i[high_temp_cond]**2
    mid_temp_cond = (temp_max_i > 0) & (temp_max_i < 26.5)
    # TODO: how to handle HEATcal: does heat change at any simulation step?
    # pe_i[mid_temp_cond] = 16 * np.power(
    #     10 * temp_max_i[mid_temp_cond] / heat[mid_temp_cond],
    #     A[mid_temp_cond])
    pe_i[mid_temp_cond] = 16 * (
        (10 *
         (temp_max_i[mid_temp_cond] / heat[mid_temp_cond]))**A[mid_temp_cond])
    pe_i[temp_max_i <= 0] = 0
    pe_i = pe_i * cropf * CROPF_COEFF

    return pe_i


def soil_storage(liquid_prec_i, pe_i, available_water_prev, whc):
    prec_eff_i = liquid_prec_i - pe_i
    # start with the water available at the end of the previous iteration
    available_water_i = np.copy(available_water_prev)
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

    return excess_i, available_water_i
