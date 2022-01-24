from endofoutbreak import data


def run_brute_force_comparison_pipeline(dataset):
    """A pipeline to compare the MCMC approach with a brute force calculation.

    The existing approximation from the literature is also included for comparison.

    Since the brute force approach is very slow, this pipelin is only intended to be run
    on very simple toy datasets. For example, where the cases day-by-day for the whole
    outbreak are [1, 0, 2, 1].

    The output is a graph comparing the end-of-outbreak date for many runs of the MCMC,
    the brute force approach and the approximate analytic formula.
    """
    params = data.load_params(dataset)
    obs_df = data.load_observations(dataset)
