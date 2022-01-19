from datetime import timedelta

import numpy as np
import scipy.stats

from endofoutbreak.estimation.misc import DiscretisedWeibull


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
