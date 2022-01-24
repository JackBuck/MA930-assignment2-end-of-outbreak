from collections import deque

import numpy as np
import scipy.stats

from endofoutbreak.estimation.misc import DiscretisedWeibull


def calculate_end_of_outbreak_probability_brute_force_estimate(
    obs_df,
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
    n_days_after_last_case_to_simulate,
    max_samples,  # TODO: Add controls to aim for a certain number of samples?
    rng=None,
):
    if not rng:
        rng = np.random.default_rng()

    cases = obs_df.set_index("time", verify_integrity=True)["num_cases"]
    cases = cases.resample("D").sum()
    first_date = cases.index[0]
    last_case_date = cases[cases > 0].index[-1]
    cases = cases[:last_case_date].tolist()

    # Set up seed infections as all infections on the first date with infections
    idx, first_case_count = next((i, n) for i, n in enumerate(cases) if n != 0)
    seed_infections = [idx] * first_case_count

    samples = []
    for _ in range(max_samples):
        new_sample = simulate_epidemic(
            seed_infections,
            r0,
            offspring_shape,
            serial_interval_mean,
            serial_interval_shape,
            n_days_to_simulate=(
                (last_case_date - first_date).days + n_days_after_last_case_to_simulate
            ),
            rng=rng,
        )
        if new_sample[:len(cases)].tolist() == cases:
            samples.append(new_sample)

    # TODO: I still need to calculate the end-of-outbreak probability at each date from
    #  this!

    return np.array(samples)


def simulate_epidemic(
    seed_infections,
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
    n_days_to_simulate,
    rng,
):
    """ Simulate an epidemic using a Branching process model

    Args:
        seed_infections (list[int]): A collection of non-negative integer seed infection
            "times" representing the days on which to simulate external infections
        r0 (float): The R-number, used to define the Negative-Binomial offspring
            distribution
        offspring_shape (float): The shape parameter for the Negative-Binomial
            distribution used to determine the number of offspring
        serial_interval_mean (float): The mean of the Weibull distribution
            characterising the serial interval (which will be discretised)
        serial_interval_shape (float): The shape parameter of the Weibull distribution
            characterising the serial interval (which will be discretised)
        n_days_to_simulate (int): The number of days for which to simulate the epidemic
        rng: A numpy random number generator

    Returns:
        np.ndarray: A 1D numpy vector counting the number of infections on each day
    """
    p = r0 / (offspring_shape + r0)
    offspring_rv = scipy.stats.nbinom(offspring_shape, 1-p)
    serial_interval_rv = DiscretisedWeibull(serial_interval_mean, serial_interval_shape)

    unprocessed_infections = deque(seed_infections)
    infections = np.zeros(n_days_to_simulate)

    while unprocessed_infections:
        t = unprocessed_infections.popleft()
        if t < n_days_to_simulate:
            infections[t] += 1

            noffspring = offspring_rv.rvs(random_state=rng)
            si = serial_interval_rv.rvs(noffspring, random_state=rng)
            unprocessed_infections.extendleft(t + si)

    return infections


########################################################################################
# Once you have the 10,000 simulations, you can use them to generate a probability of  #
# the outbreak being over at any time after the last case.                             #
# You will then be able to feed this into calculate_date_outbreak_declared_over().     #
########################################################################################


if __name__ == "__main__":
    # TODO: Temporary testing code!
    from endofoutbreak import data

    dataset = "toy-example"

    params = data.load_params(dataset)
    obs_df = data.load_observations(dataset)
    samples = calculate_end_of_outbreak_probability_brute_force_estimate(
        obs_df,
        params["offspring"]["r0"],
        params["offspring"]["shape"],
        params["serial_interval"]["mean"],
        params["serial_interval"]["shape"],
        n_days_after_last_case_to_simulate=10,
        max_samples=1000,
    )
    print(len(samples))
