import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from . import utils

__all__ = ['plot_gauge_flow']


def plot_gauge_flow(sim_gauge_flow, obs_gauge_flow=None, num_warmup_months=12,
                    warmup_vline=True, nash_sutcliffe=True,
                    monthly_aligned=True, legend=True, **plt_kws):

    fig, ax = plt.subplots(**plt_kws)

    ax.plot(sim_gauge_flow, label='simulated')

    if obs_gauge_flow is not None:
        ax.plot(obs_gauge_flow, label='real')

        if nash_sutcliffe:
            ns = utils.nash_sutcliffe(sim_gauge_flow[num_warmup_months:],
                                      obs_gauge_flow[num_warmup_months:])
            txt = "Nash-Sutcliffe"
            if num_warmup_months > 0:
                txt += f" (excluding {num_warmup_months} warm-up months)"
            txt += f": {ns}"

            ax.text(.05, .9, txt, horizontalalignment='left',
                    verticalalignment='bottom', transform=ax.transAxes)

    if monthly_aligned:
        num_years = len(sim_gauge_flow) // 12

        ax.xaxis.set_ticks(np.arange(num_years * 12))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, pos: int((x + 1) % 12) if (x + 1) % 12 != 0 else 12))

        for i in range(num_years):
            ax.axvline(x=i * 12, color='r')

        if warmup_vline:
            ax.axvline(x=num_warmup_months, color='g', linestyle='--')
            # TODO: warmup text annotation/legend

    if legend:
        ax.legend()

    ax.set_ylabel("Flow m^3/s")
    ax.set_xlabel("time (months)")

    return ax
