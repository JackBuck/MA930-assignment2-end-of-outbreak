from endofoutbreak import data
from endofoutbreak.estimation import (
    sample_end_of_outbreak_probability_via_mcmc_for_multiple_current_times,
    calculate_end_of_outbreak_probability_estimates_from_mcmc_samples,
    calculate_alternative_end_of_outbreak_probability_estimate,
    calculate_date_outbreak_declared_over,
)
from endofoutbreak.plotting import (
    plot_branching_process_distributions,
    generate_mcmc_trace_plots_for_each_time,
    plot_probability_of_future_infections,
)
from endofoutbreak.utils import extract_infection_times


def run_backtest_pipeline(
    dataset,
    *,
    plot_model_distributions=True,
    rerun_mcmc=True,
    generate_trace_plots=True,
    generate_eoo_plot=True,
):
    """ A data pipeline to 'backtest' the calculation.

    The pipeline takes the raw data and uses the MCMC method to estimate the probability
    that the outbreak is over at each time since the start of the outbreak until several
    days after the last case. At each time point, only data available at that time is
    used to make the calculation. Plots are generated comparing this estimate to the
    approximation which is currently used in the literature.

    Different stages of the calculation can be switched off and on using the keyword
    arguments to the pipeline function.
    """

    # NOTE:
    #   Some functions in this pipeline take the data as input, while others instead
    #   load and save the data within the function. The functions which load and save
    #   their own data can be identified as those accepting a `dataset` argument. The
    #   reason for allowing some functions to load and save their own data is so that
    #   the pipeline has some form of checkpointing.
    # TODO: Move to a 3rd party data pipeline library with more principled management of
    #  data.

    params = data.load_params(dataset)
    obs_df = data.load_observations(dataset)
    first_date, horizon, infection_date_offsets = extract_infection_times(obs_df)

    if plot_model_distributions:
        plot_branching_process_distributions(
            params["offspring"]["r0"],
            params["offspring"]["shape"],
            params["serial_interval"]["mean"],
            params["serial_interval"]["shape"],
            save_path=data.get_plots_filepath(dataset, "model-distributions.png"),
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
            dataset=dataset,
        )

    if generate_trace_plots:
        generate_mcmc_trace_plots_for_each_time(
            dataset, burn_in=1_000, start_date=first_date
        )

    if generate_eoo_plot:
        eoo_prob_df = calculate_end_of_outbreak_probability_estimates_from_mcmc_samples(
            first_date, dataset, burn_in=10_000
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
        save_path = data.get_plots_filepath(dataset, "eoo-plot.png")
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
