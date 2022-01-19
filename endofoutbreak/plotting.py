import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import numpy as np
import scipy.stats
from tqdm import tqdm

from endofoutbreak.data import lazy_load_samples, get_plots_directory
from endofoutbreak.estimation import estimate_end_of_outbreak_probability_from_mcmc_result, DiscretisedWeibull


def plot_adjacency_matrix(
    delay_matrix, loglikelihood, eoo_probability, accepted, fignum, show=False
):
    fig, ax = plt.subplots(constrained_layout=True, num=f"adjmat_{fignum}")
    im = ax.matshow(delay_matrix)
    accepted_str = "accepted" if accepted else "not accepted"
    ax.set_title(
        f"({fignum}) Log-likelihood: {loglikelihood:.3g} ({accepted_str})\n"
        f"Probability that the outbreak is over: {eoo_probability:.2%}"
    )
    fig.colorbar(im, label="Time between infections")
    if show:
        plt.show()
    else:
        return fig


def plot_end_of_outbreak_probability_samples(
    samples, loglikelihood, burn_in, horizon_date, show=False, save_path=None
):
    # TODO: Consider moving the responsibility for this call to the caller
    end_of_outbreak_probability = estimate_end_of_outbreak_probability_from_mcmc_result(
        samples, burn_in
    )

    fig, axs = plt.subplots(
        2, 2,
        constrained_layout=True,
        figsize=(12, 6),
        gridspec_kw=dict(
            width_ratios=(3, 1)
        ),
    )

    axs[0, 0].plot(samples, label="Sample trace")
    axs[0, 0].axhline(
        end_of_outbreak_probability,
        linestyle="dashed",
        color="red",
        label="Mean (after burn-in)",
    )
    axs[0, 0].axvspan(0, burn_in, color="grey", alpha=0.5, label="Burn-in period")
    axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axs[0, 0].set_xlabel("MCMC iteration")
    axs[0, 0].set_ylabel("Probability")
    axs[0, 0].set_title(
        "End-of-outbreak probability for sampled truncated infection processes"
    )
    axs[0, 0].legend(loc="upper right")

    axs[1, 0].plot(loglikelihood, label="Sample trace")
    axs[1, 0].axvspan(0, burn_in, color="grey", alpha=0.5, label="Burn-in period")
    axs[1, 0].set_xlabel("MCMC iteration")
    axs[1, 0].set_ylabel("Log-likelihood")
    axs[1, 0].set_title("Log-likelihood of sampled truncated infection processes")
    axs[1, 0].legend(loc="upper right")

    axs[0, 1].hist(samples[burn_in:], label="Samples")
    axs[0, 1].axvline(
        end_of_outbreak_probability, linestyle="dashed", color="red", label="Mean",
    )
    axs[0, 1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axs[0, 1].set_xlabel("Probability")
    axs[0, 1].set_ylabel("Number of observations")
    axs[0, 1].set_title(
        "End-of-outbreak probability\nhistogram (after burn-in)"
    )
    axs[0, 1].legend(loc="upper right")

    axs[1, 1].hist(loglikelihood[burn_in:])
    axs[1, 1].set_xlabel("Log-likelihood")
    axs[1, 1].set_ylabel("Number of observations")
    axs[1, 1].set_title("Log-likelihood histogram\n(after burn-in)")

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)

    if isinstance(horizon_date, datetime):
        horizon_date = horizon_date.date()
    fig.suptitle(
        f"Trace of MCMC for end-of-outbreak probabilities as of {horizon_date} "
        f"(estimate {end_of_outbreak_probability:.0%})",
        fontsize="xx-large",
    )

    if save_path:
        fig.savefig(save_path)

    if show:
        plt.show()
    else:
        return fig


def generate_mcmc_trace_plots_for_each_time(dataset, burn_in, start_date):
    plots_dir_path = get_plots_directory(dataset, "traces")

    data_loaders = lazy_load_samples(dataset)
    desc = "Generating trace plots"
    for current_time, loader in tqdm(data_loaders.items(), desc=desc):
        eoo_prob_samples = loader["eoo_probability"]()
        loglikelihoods = loader["loglikelihood"]()

        save_path = os.path.join(plots_dir_path, f"trace_{current_time}")
        fig = plot_end_of_outbreak_probability_samples(
            eoo_prob_samples,
            loglikelihoods,
            burn_in,
            horizon_date=start_date + timedelta(days=current_time),
            save_path=save_path,
        )
        plt.close(fig)


def plot_probability_of_future_infections(
    case_counts_df, eoo_prob_df, show=False, save_path=None, save_dpi=120
):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True, sharex=True)

    axs[0].plot(
        eoo_prob_df["time"],
        1 - eoo_prob_df["eoo_probability"],
        color="tab:orange",
        drawstyle="steps-mid",
        label="Our method (via McMC)",
    )
    axs[0].plot(
        eoo_prob_df["time"],
        1 - eoo_prob_df["eoo_probability_alternative"],
        color="tab:green",
        drawstyle="steps-mid",
        label="Alternative approximation",
    )
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("Probability")
    axs[0].set_title("Probability of future infections")
    axs[0].legend()

    axs[1].bar(case_counts_df["time"], case_counts_df["num_cases"], color="tab:blue")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Number of cases")
    axs[1].set_title("Observed cases")

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)

    one_day = timedelta(days=1)
    axs[0].set_xlim(
        min(case_counts_df["time"].min() - one_day, eoo_prob_df["time"].min()),
        max(case_counts_df["time"].max() + one_day, eoo_prob_df["time"].max()),
    )
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator(interval_multiples=False))

    fig.suptitle(
        "Probability of future infections conditional on observed cases to date",
        fontsize="x-large",
    )

    if save_path:
        fig.savefig(save_path, dpi=save_dpi)

    if show:
        plt.show()
    else:
        return fig


def plot_branching_process_distributions(
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
    show=False,
    save_path=None,
    save_dpi=120,
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

    # Mass function for the number of direct transmissions
    p = r0 / (r0 + offspring_shape)
    rv = scipy.stats.nbinom(offspring_shape, 1 - p)
    x = np.arange(rv.ppf(0.99))
    pmf = rv.pmf(x)
    axs[0].bar(
        x, pmf, label=rf"Negative binomial ($R_0 = {r0}$, $k={offspring_shape}$)"
    )
    axs[0].set_title("Number of onward transmissions")
    axs[0].set_xlabel("Number of direct onward transmissions")
    axs[0].set_ylabel("Probability")
    axs[0].legend(loc="upper right")

    serial_interval_rv = DiscretisedWeibull(serial_interval_mean, serial_interval_shape)
    t_discr = np.arange(1, serial_interval_rv.continuous_rv.ppf(0.99) + 1, dtype="int")
    serial_interval_pmf = serial_interval_rv.pmf(t_discr)
    axs[1].bar(t_discr, serial_interval_pmf, label="Discrete approximation")
    t = np.linspace(0, serial_interval_rv.continuous_rv.ppf(0.99), 1000)
    serial_interval_cts_pdf = serial_interval_rv.continuous_rv.pdf(t)
    axs[1].plot(
        t,
        serial_interval_cts_pdf,
        label=rf"Weibull ($\mu={serial_interval_mean}$, $k={serial_interval_shape}$)",
        color="tab:green",
    )
    axs[1].set_title("Serial interval")
    axs[1].set_xlabel("Serial interval / days")
    axs[1].set_ylabel("Probability")
    axs[1].legend(loc="upper left", bbox_to_anchor=(0.6, 1))

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Probability mass functions used in branching process model",
        fontsize="xx-large",
    )

    if save_path:
        fig.savefig(save_path, dpi=save_dpi)

    if show:
        plt.show()
    else:
        return fig
