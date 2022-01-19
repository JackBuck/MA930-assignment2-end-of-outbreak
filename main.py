import os
from datetime import timedelta

import dateutil.parser
import numpy as np
import pandas as pd

from endofoutbreak.data import DATA_DIR, load_params
from endofoutbreak.estimation import (
    sample_end_of_outbreak_probability_via_mcmc_for_multiple_current_times,
    calculate_end_of_outbreak_probability_estimates,
    calculate_alternative_end_of_outbreak_probability_estimate, calculate_date_outbreak_declared_over,
)
from endofoutbreak.plotting import generate_mcmc_trace_plots_for_each_time, plot_probability_of_future_infections, \
    plot_branching_process_distributions

# The file path containing the data from the hospital related outbreak in Taiwan
DATA_FILE_NAME = "EndOfOutbreak_Taiwan_Cases_Data.csv"


def load_observations(fpath=None):
    if not fpath:
        fpath = os.path.join(DATA_DIR, DATA_FILE_NAME)

    obs_df = pd.read_csv(fpath, parse_dates=["Date"], date_parser=dateutil.parser.parse)
    obs_df = obs_df.rename(columns={"Date": "time", "Cases": "num_cases"})
    return obs_df


def extract_infection_times(obs_df):
    first_date = obs_df["time"].min()
    last_date = obs_df["time"].max()

    offsets = []
    for row in obs_df.itertuples(index=False):
        for _ in range(row.num_cases):
            offsets.append(int((row.time - first_date) / timedelta(days=1)))

    offsets = sorted(offsets)
    offsets = np.asarray(offsets)

    last_possible_offset = int((last_date - first_date) / timedelta(days=1))

    return first_date, last_possible_offset, offsets


def main(
    plot_model_distributions=True,
    rerun_mcmc=True,
    generate_trace_plots=True,
    generate_eoo_plot=True,
):
    """ An ad-hoc data pipeline """
    dataset = "taiwan"
    params = load_params(dataset)
    obs_df = load_observations()
    first_date, horizon, infection_date_offsets = extract_infection_times(obs_df)

    if plot_model_distributions:
        plot_branching_process_distributions(
            params["offspring"]["r0"],
            params["offspring"]["shape"],
            params["serial_interval"]["mean"],
            params["serial_interval"]["shape"],
            save_path=os.path.join(DATA_DIR, "plots", "taiwan-model-distributions.png"),
            show=True,
        )

    if rerun_mcmc:
        sample_end_of_outbreak_probability_via_mcmc_for_multiple_current_times(
            infection_date_offsets,
            params["offspring"]["r0"],
            params["offspring"]["shape"],
            params["serial_interval"]["mean"],
            params["serial_interval"]["shape"],
            nsteps=10_000,
            subdirectory=dataset,
        )

    if generate_trace_plots:
        generate_mcmc_trace_plots_for_each_time(
            subdirectory=dataset, burn_in=1_000, start_date=first_date
        )

    if generate_eoo_plot:
        eoo_prob_df = calculate_end_of_outbreak_probability_estimates(
            first_date, subdirectory=dataset, burn_in=10_000
        )
        eoo_prob_df["eoo_probability_alternative"] = calculate_alternative_end_of_outbreak_probability_estimate(
            eoo_prob_df["time"],
            first_date,
            infection_date_offsets,
            params["offspring"]["r0"],
            params["offspring"]["shape"],
            params["serial_interval"]["mean"],
            params["serial_interval"]["shape"],
        )

        save_path = os.path.join(DATA_DIR, "plots", f"{dataset}-eoo-plot.png")
        plot_probability_of_future_infections(
            obs_df, eoo_prob_df, show=True, save_path=save_path
        )

        outbreak_over = calculate_date_outbreak_declared_over(obs_df, eoo_prob_df)
        print(
            "Dates at the end of which the outbreak would be declared over:\n"
            "  Our method: {} ({} days after last case)\n"
            "  Alternative method: {} ({} days after last case)".format(
                outbreak_over["date_declared_over"].date(),
                outbreak_over["days_after_last_case"],
                outbreak_over["date_declared_over_alt"].date(),
                outbreak_over["days_after_last_case_alt"],
            )
        )


if __name__ == "__main__":
    main(
        plot_model_distributions=False,
        rerun_mcmc=False,
        generate_trace_plots=True,
        generate_eoo_plot=True,
    )
