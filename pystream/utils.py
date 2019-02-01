import numpy as np

__all__ = ['nash_sutcliffe']


def nash_sutcliffe(sim_gauge_flow, obs_gauge_flow):
    if len(sim_gauge_flow) != len(obs_gauge_flow):
        raise ValueError("Lengths must match!")

    return 1 - sum((sim_gauge_flow - obs_gauge_flow)**2) / sum(
        (obs_gauge_flow - np.mean(obs_gauge_flow))**2)
