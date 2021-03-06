import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from endofoutbreak import data
from endofoutbreak.estimation.misc import DiscretisedWeibull


def sample_end_of_outbreak_probability_via_mcmc_for_multiple_current_times(
    infection_times,
    r0,
    offspring_shape,
    serial_interval_mean,
    serial_interval_shape,
    nsteps,
    dataset,
    initial_delay_matrix=None,
    min_current_time=None,
    max_current_time=None,
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
        data.save_samples(eoo_probability, loglikelihood, dataset, suffix=current_time)


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


def calculate_end_of_outbreak_probability_estimates_from_mcmc_samples(
    first_date, dataset, burn_in
):
    data_loaders = data.lazy_load_samples(dataset)

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


def estimate_end_of_outbreak_probability_from_mcmc_result(
    samples, burn_in, thinning=1
):
    """ Estimate the probability of no more infections from the MCMC result """
    return np.mean(samples[burn_in::thinning])
