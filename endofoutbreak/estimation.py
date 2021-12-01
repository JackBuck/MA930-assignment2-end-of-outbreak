import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from tqdm import tqdm

from endofoutbreak.data import save_samples, lazy_load_samples


class DiscretisedWeibull:
    def __init__(self, mean, shape):
        self._mean = mean
        self._shape = shape
        self.continuous_rv = self._make_cts_rv(mean, shape)

    @staticmethod
    def _make_cts_rv(mean, shape):
        weibull_scale = mean / scipy.special.gamma(1 + 1 / shape)
        return scipy.stats.weibull_min(shape, scale=weibull_scale)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, new_mean):
        self._mean = new_mean
        self.continuous_rv = self._make_cts_rv(new_mean, self._shape)

    def pmf(self, x):
        """Probability mass function"""
        x = np.asarray(x)
        if x.dtype != "int":
            raise TypeError(
                f"Expected an integer or integer array. Got {x!r} of type"
                f"type {type(x)} and dtype {x.dtype}."
            )

        # Where x=1 we want to integrate from 0 to 1.5; when x=0 we want to return 0.
        cdf_hi = self.continuous_rv.cdf(np.where(x > 0, x + 0.5, 0))
        cdf_lo = self.continuous_rv.cdf(np.where(x > 1, x - 0.5, 0))
        probs = cdf_hi - cdf_lo

        return probs

    def cdf(self, x):
        """Cumulative distribution function"""
        x = np.asarray(x)
        if x.dtype != "int":
            raise TypeError(
                f"Expected an integer or integer array. Got {x!r} of type"
                f"type {type(x)} and dtype {x.dtype}."
            )

        cdf = self.continuous_rv.cdf(np.where(x > 0, x + 0.5, 0))
        return cdf

    def sf(self, x):
        """Survival function (1-cdf)"""
        return 1 - self.cdf(x)


def estimate_end_of_outbreak_probability_from_mcmc_result(
    samples, burn_in, thinning=1
):
    """ Estimate the probability of no more infections from the MCMC result """
    return np.mean(samples[burn_in::thinning])


def sample_end_of_outbreak_probability_via_mcmc_for_multiple_current_times(
    infection_times,
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
    nsteps,
    initial_delay_matrix=None,
    min_current_time=None,
    max_current_time=None,
    subdirectory=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if min_current_time is None:
        if initial_delay_matrix:
            is_external = (initial_delay_matrix > 0).sum(axis=1)
            [external_cases] = is_external.nonzero()
            min_current_time = external_cases.max()
        else:
            min_current_time = infection_times.min()

    if max_current_time is None:
        max_current_time = infection_times.max() + 10

    desc = "Backtesting"
    for current_time in tqdm(range(min_current_time, max_current_time+1), desc=desc):
        is_in_past = infection_times <= current_time
        if initial_delay_matrix:
            # TODO: Check that the external cases are all in the past...
            initial_delay_matrix_trunc = initial_delay_matrix[is_in_past, is_in_past]
        else:
            initial_delay_matrix_trunc = None

        eoo_probability, loglikelihood = sample_end_of_outbreak_probability_via_mcmc(
            infection_times[is_in_past],
            r0,
            offspring_shape,
            serial_interval_mean,
            serial_interval_shape,
            current_time,
            nsteps,
            initial_delay_matrix_trunc,
            copy.copy(rng),
        )
        save_samples(eoo_probability, loglikelihood, subdirectory, suffix=current_time)


def sample_end_of_outbreak_probability_via_mcmc(
    infection_times,
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
    current_time,
    nsteps,
    initial_delay_matrix=None,
    rng=None,
):
    """
    Sample the probability of no more infections conditional on observations via MCMC.

    Observations here means a list of infection times (integers representing days since
    the first infection). Days with multiple infections should already be expanded in
    this array. The array should be sorted in increasing order.

    No burn-in or thinning is applied to the result before returning it. This must be
    done by the caller.

    Args:
        infection_times: An increasing list of infection times, starting with the first
            observed infection (usually at time 0)
        r0: The R-number, used to define the Negative-Binomial offspring distribution
        offspring_shape: The shape parameter for the Negative-Binomial distribution used
            to determine the number of offspring
        serial_interval_mean: The mean of the Weibull distribution characterising the
            serial interval (which will be discretised)
        serial_interval_shape: The shape parameter of the Weibull distribution
            characterising the serial interval (which will be discretised)
        current_time: The latest day on which we have an observation (should be at least
            as late as the latest observed infection, but could be greater if there are
            days observed to have no infections)
        initial_delay_matrix: The adjacency matrix for the infection graph at which to
            start the MCMC, weighted by the serial intervals between infections
        nsteps: The number of iterations to make of the Markov chain
        rng: A numpy random number generator

    Returns:
        (eoo_probability, loglikelihood) - the vector of end-of-outbreak probabilities
            and the vector of corresponding log-likelihoods.
    """
    # TODO: Multiple chains vectorised
    # TODO: Multiple chains via multiprocessing & concurrent.futures

    if not rng:
        rng = np.random.default_rng()

    if any(t > current_time for t in infection_times):
        raise ValueError(
            f"Expected all observed infections to happen before current_time. "
            f"Got {current_time=}, {max(infection_times)=}."
        )

    serial_interval_rv = DiscretisedWeibull(serial_interval_mean, serial_interval_shape)

    if initial_delay_matrix:
        delay_matrix = initial_delay_matrix
    else:
        delay_matrix = generate_initial_delay_matrix(
            infection_times, serial_interval_rv, rng=rng
        )

    eoo_probability = np.zeros(nsteps + 1)
    eoo_probability[0] = calc_conditional_prob_of_no_more_infections(
        delay_matrix,
        r0,
        offspring_shape,
        serial_interval_rv,
        infection_times,
        current_time,
    )

    loglikelihood = np.zeros(nsteps + 1)
    loglikelihood[0] = calc_truncated_infection_process_loglikelihood(
        delay_matrix,
        r0,
        offspring_shape,
        serial_interval_rv,
        infection_times,
        current_time,
    )

    desc = "Running MCMC"
    for i in tqdm(range(nsteps), leave=False, desc=desc):
        candidate, proposal_prob_ratio = propose_new_infection_process(
            delay_matrix, infection_times, serial_interval_rv, rng
        )
        candidate_ll = calc_truncated_infection_process_loglikelihood(
            candidate,
            r0,
            offspring_shape,
            serial_interval_rv,
            infection_times,
            current_time,
        )

        # The proposal distribution is symmetric, so we can drop it from the formula in
        # the Hastings algorithm. Therefore, the log of the acceptance probability is
        # the smaller of 0 and candidate_ll - loglikelihood[i]
        u = rng.uniform()
        if np.log(u) < candidate_ll - loglikelihood[i] + np.log(proposal_prob_ratio):
            delay_matrix = candidate
            eoo_probability[i+1] = calc_conditional_prob_of_no_more_infections(
                delay_matrix,
                r0,
                offspring_shape,
                serial_interval_rv,
                infection_times,
                current_time,
            )
            loglikelihood[i+1] = candidate_ll

        else:
            eoo_probability[i+1] = eoo_probability[i]
            loglikelihood[i+1] = loglikelihood[i]

    return eoo_probability, loglikelihood


def generate_initial_delay_matrix(
    infection_times,
    serial_interval_rv,
    external_prob=0,
    *,
    rng,
):
    """ Generate an initial infection process consistent with the observations

    Each infection is assigned a parent infection from among those strictly preceding
    it, with probability proportional to the serial interval pmf (but independent of the
    number of infections currently caused by the prospective parent). Further, there is
    a probability external_prob that each infection is an external infection. Infections
    observed on the first day of infections are always external infections.

    Args:
        infection_times: An increasing list of infection times, starting with the first
            observed infection (usually at time 0)
        serial_interval_rv: An object representing the (discrete) serial interval
            distribution, with support on the (strictly) positive integers
        external_prob: The probability that any given individual is an external
            infection
        rng: A numpy random number generator

    Returns:
        The delay_matrix for the random initial infection process
    """
    nindividuals = len(infection_times)
    infection_times = np.asarray(infection_times)

    # Element (i,j) is t[i]-t[j]
    time_differences = infection_times[:, np.newaxis] - infection_times[np.newaxis, :]

    # Select the "parent" of each infection from the earlier infections with probability
    # proportional to the serial interval pmf.
    # The (i,j)th element of `probs` is the probability that individual i was infected
    # by individual j.
    probs = serial_interval_rv.pmf(time_differences)
    rowsums = probs.sum(axis=1, keepdims=True)
    probs /= np.where(rowsums > 0, rowsums, 1)
    # We append a column at the end here to represent the probability of an external
    # infection. If all probabilities are zero then it must be an external infection
    # with probability 1 (this happens for the first infection case, as well as if all
    # previous infections are too long ago to be possible parents).
    probs = np.append(
        probs * (1 - external_prob),
        np.where(rowsums > 0, external_prob, 1),
        axis=1,
    )

    parent_infections = [rng.choice(nindividuals+1, p=p) for p in probs]

    delay_matrix = np.zeros((nindividuals, nindividuals), dtype="int")
    for i, parent in enumerate(parent_infections):
        if parent <= nindividuals - 1:
            delay_matrix[parent, i] = time_differences[i, parent]
        # else: External infection => Leave all entries in column i as zero

    return delay_matrix


def calc_conditional_prob_of_no_more_infections(
    delay_matrix,
    r0,
    offspring_shape,
    serial_interval_rv,
    infection_times,
    current_time,
):
    """
    Calculate the probability of no more infections given the infection process so far

    Args:
        delay_matrix: The adjacency matrix for the infection graph, weighted by the
            serial intervals between infections
        r0: The R-number, used to define the Negative-Binomial offspring distribution
        offspring_shape: The shape parameter for the Negative-Binomial distribution used
            to determine the number of offspring
        serial_interval_rv: An object representing the (discrete) serial interval
            distribution, with support on the (strictly) positive integers
        infection_times: The vector of infection times
        current_time: The latest day on which we have an observation (should be at least
            as late as the latest observed infection, but could be greater if there are
            days observed to have no infections)

    Returns:
        The conditional probability that there will be no more infections given the full
            infection process to date.
    """

    p = r0 / (offspring_shape + r0)
    num_onward_infections = np.count_nonzero(delay_matrix, axis=1)
    time_since_infection = current_time - infection_times

    # Calculate the probability that a given serial interval is greater than (>) the
    # current_time - infection_times. I.e. that the corresponding infection occurs after
    # our observation window.
    si_tail_by_infection = serial_interval_rv.sf(time_since_infection)

    # Calculate the probability of no more infections using the formula calculated in
    # the report
    q = si_tail_by_infection
    e = num_onward_infections
    k = offspring_shape
    prob_of_no_more_infections = np.prod((1 - p * q) ** (e + k))

    return prob_of_no_more_infections


def propose_new_infection_process(
    current_delay_matrix, infection_times, serial_interval_rv, rng
):
    """ Propose a new infection process for the MCMC.

    A new proposal is made by randomly selecting an infection whose parent to change,
    and then randomly selecting a new parent uniformly from the other options.

    Args:
        current_delay_matrix: The delay matrix for the current infection process
        infection_times: The vector of infection times
        serial_interval_rv: An object representing the (discrete) serial interval
            distribution, with support on the (strictly) positive integers
        rng: A numpy random number generator

    The proposed process will be consistent with the observed infection times, and will
    preserve the set of external infections in the current infection process.

    Proposal probabilities are NOT symmetric, in the sense that proposing process B when
    currently at process A has a different probability as proposing A when currently at
    B. This means that we need to account for the proposal probabilities when
    determining our acceptance probabilities in the Hasting's algorithm. To this end,
    the ratio of the proposal probabilities, h(current|proposal)/h(proposal|current), is
    returned alongside the proposed delay_matrix.

    Returns:
        Tuple (delay_matrix, proposal_probability_ratio)
            - `delay_matrix` is the delay matrix of the new infection process
            - `proposal_probability_ratio` is the ratio of the backward and forward
                proposal probabilities required in the Hastings MCMC algorithm when the
                proposal distribution is not symmetric
    """
    nindividuals = len(infection_times)
    infection_times = np.asarray(infection_times)

    # The probability mass of each possible serial interval will be used throughout this
    # function, so calculate it up front
    time_differences = infection_times[:, np.newaxis] - infection_times[np.newaxis, :]
    si_probs = serial_interval_rv.pmf(time_differences)

    # Element (i, j) of is_possible_parent is True if j is a possible parent of i
    # We will use this to help select which node's parent to change
    is_possible_parent = si_probs > 0
    # Remove from each list of possible parents, the current parent (presuming it is in
    # the list)
    is_possible_parent &= (current_delay_matrix.T == 0)
    # External infections have no possible parents
    is_external = (current_delay_matrix > 0).sum(axis=0) == 0
    is_possible_parent &= ~is_external[np.newaxis, :]

    # Choose node whose parent to change and what to change it to.
    # The target node is chosen uniformly. The new parent is chosen with probabilities
    # proportional to the probability of observing the implied serial interval.
    num_possible_parents = is_possible_parent.sum(axis=1)
    if (num_possible_parents == 0).all():
        return current_delay_matrix, 1

    target_node = rng.choice(*(num_possible_parents > 0).nonzero())

    [current_parent] = current_delay_matrix[:, target_node].nonzero()
    parent_probs = si_probs[target_node].copy()
    parent_probs[current_parent] = 0
    parent_probs /= parent_probs.sum()
    new_parent = rng.choice(nindividuals, p=parent_probs)

    # Calculate the proposal probability ratio needed because the proposal distribution
    # is not symmetric
    parent_probs_back = si_probs[target_node].copy()
    parent_probs_back[new_parent] = 0
    parent_probs_back /= parent_probs_back.sum()

    p_forward = parent_probs[new_parent]
    p_backward = parent_probs_back[current_parent]
    proposal_probability_ratio = p_backward / p_forward

    # Make new delay matrix
    new_delay_matrix = current_delay_matrix.copy()
    new_delay_matrix[:, target_node] = 0
    new_delay_matrix[new_parent, target_node] = (
        infection_times[target_node] - infection_times[new_parent]
    )

    return new_delay_matrix, proposal_probability_ratio


def calc_truncated_infection_process_loglikelihood(
    delay_matrix,
    r0,
    offspring_shape,
    serial_interval_rv,
    infection_times,
    current_time,
):
    """ Calculate the likelihood of a truncated infection process.

    Args:
        delay_matrix: The adjacency matrix for the infection graph, weighted by the
            serial intervals between infections
        r0: The R-number, used to define the Negative-Binomial offspring distribution
        offspring_shape: The shape parameter for the Negative-Binomial distribution used
            to determine the number of offspring
        serial_interval_rv: An object representing the (discrete) serial interval
            distribution, with support on the (strictly) positive integers
        infection_times: The vector of infection times
        current_time: The current time (should be at least as late as the latest
            observed infection)

    Returns:
        The log-likelihood of the truncated infection process
    """

    # We assume that the delay_matrix encoding the truncated infection process is
    # consistent with the observations. Hence, the only relevant term is the probability
    # of the infection process given the external cases.

    p = r0 / (offspring_shape + r0)
    num_onward_infections = np.count_nonzero(delay_matrix, axis=1)
    time_since_infection = current_time - infection_times

    serial_interval_probs = serial_interval_rv.pmf(delay_matrix[delay_matrix > 0])
    si_tail_by_infection = serial_interval_rv.sf(time_since_infection)

    probability_of_exactly_observed_onward_infections = (
        scipy.stats.nbinom.pmf(num_onward_infections, offspring_shape, 1 - p)
    )

    # Calculate the log-likelihood of the truncated infection process using the formula
    # calculated in the report
    q = si_tail_by_infection
    e = num_onward_infections
    k = offspring_shape
    z = probability_of_exactly_observed_onward_infections
    s = serial_interval_probs

    loglikelihood = np.sum(np.log(s)) + np.sum(np.log(z) - (e+k)*np.log(1-p*q))

    return loglikelihood


def calculate_end_of_outbreak_probability_estimates(first_date, subdirectory, burn_in):
    data_loaders = lazy_load_samples(subdirectory)

    eoo_probabilities = np.zeros(len(data_loaders))
    desc = "Processing data"
    for i, (current_time, loader) in enumerate(tqdm(data_loaders.items(), desc=desc)):
        eoo_prob_samples = loader["eoo_probability"]()
        eoo_probabilities[i] = estimate_end_of_outbreak_probability_from_mcmc_result(
            eoo_prob_samples, burn_in
        )

    df = pd.DataFrame(
        {
            "time": [first_date + timedelta(days=t) for t in data_loaders],
            "eoo_probability": eoo_probabilities,
        }
    )
    return df


def calculate_alternative_end_of_outbreak_probability_estimate(
    target_dates,
    first_date,
    infection_date_offsets,
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
):
    """
    Calculate the approximation of EOO probability from current literature.

    This approximation is theoretically an over-estimation.

    This is the approximation present in:
      - Linton, N. M., Akhmetzhanov, A. R., & Nishiura, H. (2021). Localized
        end-of-outbreak determination for coronavirus disease 2019 (COVID-19): examples
        from clusters in Japan.
        International Journal of Infectious Diseases, 105, 286–292.
        https://doi.org/10.1016/j.ijid.2021.02.106
      - Akhmetzhanov, A. R., Jung, S., Cheng, H.-Y., & Thompson, R. N. (2021). A
        hospital-related outbreak of SARS-CoV-2 associated with variant Epsilon
        (B.1.429) in Taiwan: transmission potential and outbreak containment under
        intensified contact tracing, January – February 2021.
        International Journal of Infectious Diseases, 110, 15–20.
        https://doi.org/10.1016/j.ijid.2021.06.028
    """
    p = r0 / (r0 + offspring_shape)
    offspring_rv = scipy.stats.nbinom(offspring_shape, 1-p)
    serial_interval_rv = DiscretisedWeibull(serial_interval_mean, serial_interval_shape)

    eoo_probabilities = -1 * np.ones(len(target_dates))
    for i, current_date in enumerate(target_dates):
        offset = int((current_date - first_date) / timedelta(days=1))
        relevant_infection_date_offsets = infection_date_offsets[
            infection_date_offsets <= offset
        ]
        days_since_infection = offset - relevant_infection_date_offsets

        si_cdf = serial_interval_rv.cdf(days_since_infection)

        ncases = len(relevant_infection_date_offsets)
        y = np.arange(offspring_rv.ppf(0.999 ** (1 / ncases)))
        offspring_pmf = offspring_rv.pmf(y)

        terms = offspring_pmf.reshape(-1, 1) * si_cdf**y.reshape(-1, 1)
        eoo_prob = np.prod(np.sum(terms, axis=0), axis=0)
        eoo_probabilities[i] = eoo_prob

    return eoo_probabilities


def calculate_date_outbreak_declared_over(case_counts_df, eoo_prob_df, thresh=0.95):
    has_cases = case_counts_df["num_cases"] > 0
    date_of_last_case = case_counts_df.loc[has_cases, "time"].max()

    is_declared_over = eoo_prob_df["eoo_probability"] >= thresh
    is_declared_over_alt = eoo_prob_df["eoo_probability_alternative"] >= thresh
    date_declared_over = eoo_prob_df.loc[is_declared_over, "time"].min()
    date_declared_over_alt = eoo_prob_df.loc[is_declared_over_alt, "time"].min()
    days_after_last_case = (date_declared_over - date_of_last_case).days
    days_after_last_case_alt = (date_declared_over_alt - date_of_last_case).days

    return {
        "date_declared_over": date_declared_over,
        "date_declared_over_alt": date_declared_over_alt,
        "days_after_last_case": days_after_last_case,
        "days_after_last_case_alt": days_after_last_case_alt,
    }
