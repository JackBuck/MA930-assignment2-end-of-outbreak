import numpy as np
import scipy.special
import scipy.stats


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


# TODO: Extend this method to work with any set of eoo probability columns
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
