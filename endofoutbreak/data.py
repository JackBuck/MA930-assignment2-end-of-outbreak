import os
import re

import dateutil.parser
import numpy as np
import pandas as pd
import yaml


DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def load_params(dataset):
    """Load a parameters yaml file.

    The file is expected to have the following form:

        offspring:
          r0: 1.3
          shape: 0.38
        serial_interval:
          mean: 4.8
          shape: 2.3
    """
    fpath = os.path.join(DATA_DIR, "01-raw", dataset, "parameters.yaml")
    with open(fpath) as f:
        params = yaml.safe_load(f)
    return params


def load_observations(dataset=None, fpath=None):
    """Load the observations csv file"""
    if not fpath:
        if not dataset:
            raise ValueError("Must supply at least one of 'dataset' and 'fpath'")
        fpath = os.path.join(DATA_DIR, "01-raw", dataset, "case-counts.csv")

    obs_df = pd.read_csv(fpath, parse_dates=["Date"], date_parser=dateutil.parser.parse)
    obs_df = obs_df.rename(columns={"Date": "time", "Cases": "num_cases"})
    return obs_df


def save_samples(eoo_prob_samples, loglikelihoods, subdirectory=None, suffix=None):
    """ Save MCMC samples in a standard location with an optional subdirectory.

    Samples are saved using ``numpy.save()``.
    """
    path_parts = [DATA_DIR, "mcmc-samples"]
    if subdirectory:
        path_parts.append(subdirectory)
    dir_path = os.path.join(*path_parts)

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    suffix = f"_{suffix}" if suffix is not None else ""
    np.save(os.path.join(dir_path, f"eoo_prob_samples{suffix}.npy"), eoo_prob_samples)
    np.save(os.path.join(dir_path, f"loglikelihoods{suffix}.npy"), loglikelihoods)


def lazy_load_samples(subdirectory=None):
    """ Return a dictionary of functions to load data from an mcmc run.

    The returned dictionary has keys for each current_time (inferred from the file
    name). The values are themselves dictionaries of the form
    {
        "eoo_probability": <function-to-load-numpy array>,
        "loglikelihoods": <function-to-load-numpy array>,
    }
    """

    path_parts = [DATA_DIR, "mcmc-samples"]
    if subdirectory:
        path_parts.append(subdirectory)
    dir_path = os.path.join(*path_parts)

    filenames = os.listdir(dir_path)
    eoo_prob_sample_fnames = {
        int(re.match(r"^eoo_prob_samples_(\d+)\.npy$", fname).group(1)): fname
        for fname in filenames
        if fname.startswith("eoo_prob_samples_")
    }
    loglikelihood_fnames = {
        int(re.match(r"^loglikelihoods_(\d+)\.npy$", fname).group(1)): fname
        for fname in filenames
        if fname.startswith("loglikelihoods_")
    }

    if eoo_prob_sample_fnames.keys() != loglikelihood_fnames.keys():
        raise ValueError("Inconsistent keys in cached MCMC results.")
    all_current_times = sorted(eoo_prob_sample_fnames.keys())

    def make_loader(fpath):
        return lambda: np.load(fpath)

    load_functions = {}
    for current_time in all_current_times:
        eoo_prob_fpath = os.path.join(dir_path, eoo_prob_sample_fnames[current_time])
        loglikelihood_fpath = os.path.join(dir_path, loglikelihood_fnames[current_time])

        load_functions[current_time] = {
            "eoo_probability": make_loader(eoo_prob_fpath),
            "loglikelihood": make_loader(loglikelihood_fpath)
        }

    return load_functions


def get_plots_directory(dataset, subdirectory=None):
    """ Return a subdirectory in which to save plots.

    If an optional subdirectory is provided then also ensure that it exists.

    We relinquish control of saving the plot since it is more convenient to be able to
    save and then show the plot directly from the function which creates it.
    """
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Expected data directory to already exist: {DATA_DIR!r}"
        )
    plots_dir_path = os.path.join(DATA_DIR, "plots", dataset)
    if subdirectory:
        plots_dir_path = os.path.join(plots_dir_path, subdirectory)

    os.makedirs(plots_dir_path, 0x755, exist_ok=True)

    return plots_dir_path


def get_plots_filepath(dataset, *parts):
    """Return a filepath at which a plot may be saved.

    The function will ensure all subdirectories between DATA_DIR and the target
    directory exist.
    """

    dir_path = get_plots_directory(dataset, *parts[:-1])
    fpath = os.path.join(dir_path, parts[-1])

    return fpath
