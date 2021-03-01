import numpy as np


# Functions to return probabilistic variables in suitable format
def gamma(alpha, beta, sims):
    alpha = np.array([alpha] * sims)
    beta = np.array([beta] * sims)
    samples = np.random.gamma(alpha, beta)
    return samples


def gamma_specified(min, multiplier, alpha, beta, sims):
    min = np.array([min] * sims).T
    alpha = np.array([alpha] * sims)
    beta = np.array([beta] * sims)
    samples = min + np.random.gamma(alpha, beta) * multiplier
    samples = samples.T
    return samples


def normal(parameter, sd, sims):
    samples = np.random.normal(parameter, sd, (sims, 45))
    return samples


def lognormal(parameter, sd, sims):
    samples = np.random.lognormal(parameter, sd, (sims, 45))
    return samples


def beta(parameter, se, sims):
    alpha = np.array([parameter * ((parameter*(1-parameter))/(se**2)-1)] * sims)
    beta = (alpha/parameter) - alpha
    samples = np.random.beta(alpha, beta)
    samples = samples.T
    return samples


# Function to deliver PSA simulation matrix for variables not being varied
def psa_function(var, sims):
    return np.array([var] * sims)


def mean(parameter):
    return np.mean(parameter, axis=0)


def total(parameter):
    return np.sum(np.mean(parameter, axis=0))
