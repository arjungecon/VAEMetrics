import numpy as np
import pandas as pd
from numba import njit, jit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import statsmodels.api as sm
from scipy.stats import norm, zscore
import scipy as sp
from numpy import random, linalg
from scipy import sparse, stats
import itertools as it
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.linear_model import Lasso
import copy

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amssymb}',
    r'\usepackage{amsmath}',
    r'\usepackage{xcolor}',
    r'\renewcommand*\familydefault{\sfdefault}']
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['pgf.preamble'] = [
    r'\usepackage[utf8x]{inputenc}',
    r'\usepackage{amssymb}',
    r'\usepackage[T1]{fontenc}',
    r'\usepackage{amsmath}',
    r'\usepackage{sansmath}']

inv, ax, norm = np.linalg.inv, np.newaxis, np.linalg.norm
randint = np.random.randint


def lasso_objective(b, y, X, lmbda):

    # Question 1 Part A

    """
        Function that accepts the guess for the parameter vector and LASSO penalty multipler λ,
         and computes the LASSO objective function based on the input data.
        :param b: Parameter vector.
        :param y: Outcome variable, vector of size N.
        :param X: Covariate variables (may or may not include ι), matrix of size N x P.
        :param lmbda: LASSO penalty.
        :return: Objective function evaluated using inputs.
    """

    # Return the objective function if matrix multiplication Xβ is compatible.
    try:
        obj = np.square(y - X @ b).sum()/2 + lmbda * norm(b, ord=1)
        return obj
    except:
        print("Error: The number of covariates is not compatible with given coefficient vector.")
        return np.inf


def dual_sol(bj, lmbda):

    """
        Function that returns the solution for a single coordinate in the Cyclic
         Coordinate Descent algorithm given the OLS coordinate estimate and the
         LASSO penalty multipler.
        :param bj: OLS estimate for coordinate j.
        :param lmbda: LASSO penalty multiplier.
        :return: LASSO coordinate estimate.
    """

    if bj < - lmbda:
        return bj + lmbda
    elif bj > lmbda:
        return bj - lmbda
    else:
        return 0


def lambda_zero(y, X, standardized=False):

    # Question 1 Part D

    """
        Function that computes the smallest penalty at which the LASSO estimate is exactly equal to zero.
        :param y: Outcome variable, vector of size N.
        :param X: Covariate variables (may or may not include ι), matrix of size N x P.
        :param standardized: Indicator for whether the data has been standardized.
        :return: The lambda penalty.
    """

    if standardized is False:
        X, y = zscore(X, axis=0), zscore(y)

    lmbda_max = np.max(X.T @ y)

    return lmbda_max


def lasso_cdg(b_start, y, X, lmbda, eps=1e-6, max_iter=1000, standardized=False, 
              active_set=False, active_set_cycle=10, safe=False):

    # Question 1 Part B + C + E

    """
        Function that performs the LASSO estimation through the Cyclic Coordinate Descent algorithm. Additional
         options for using the Active Set Strategy and the SAFE algorithm are provided.

        :param b_start: Initial guess for the parameter vector (may or may not include b0, which will be 
         trimmed out if so)
        :param y: Outcome variable, vector of size N.
        :param X: Covariate variables (may or may not include ι), matrix of size N x P.
        :param lmbda: LASSO penalty multiplier.
        :param eps: Norm stopping criterion.
        :param max_iter: Iteration number stopping criterion.
        :param standardized: Indicator for whether the data has been standardized.

        :param active_set: Option for using the Active Set strategy to increase the speed of the algorithm.
        :param active_set_cycle: The frequency at which all covariates are updated in CDG rather than those in
         the Active Set. This option is not used if "active_set" is False.
        :param safe: Option for using the SAFE strategy to discard covariates.

        :return: List containing:
            - :estimate: final coefficient vector estimate,
            - :objectives: vector containing LASSO objective function values
            - :steps: vector containing norm of difference in estimated parameter vectors
            - :status: string regarding which stopping criterion was used.
    """

    p, N, b_guess = b_start.size, y.size, b_start

    if N != X.shape[0]:
        print("Error: Covariate matrix is incompatible with outcome variable.")
        return None
    elif p != X.shape[1]:
        print("Error: Covariate matrix is incompatible with parameter vector.")
        return None

    # Detect if a constant term is included
    iota = (X[:, 0] == X[:, 0].mean()).all()

    # Trim out constant term
    if iota:
        X, b_guess = X[:, 1:], b_guess[1:]
        p = p - 1

    # Standardize data if not done so
    if standardized is False:
        X_mean, y_mean = X.mean(axis=0), y.mean()
        X_std, y_std = X.std(axis=0), y.std()
        X, y = zscore(X, axis=0), zscore(y)

    # LASSO objective
    lasso_obj = lambda b: lasso_objective(b, y, X, lmbda)

    # Default options for Active Set and SAFE strategies
    active_range = lambda x: np.arange(0, p)
    safe_range = np.arange(0, p)

    # Implementing Active Set Strategy
    if active_set:
        active_range = lambda x: np.where(x != 0)[0]

    # Implementing SAFE Strategy
    if safe:
        lmbda_max = lambda_zero(y=y, X=X, standardized=True)

        safe_range = np.where((X.T @ y).squeeze() >= lmbda_max - norm(X, axis=1) * norm(y) *
                              (lmbda_max - lmbda)/lmbda)

    keyDict = {"estimate", "objectives", "steps", "status"}
    output = dict([(key, []) for key in keyDict])

    niter, dist = 0, 1000

    # While loop to perform LASSO minimization using two stopping criterion.
    while niter < max_iter and dist > eps:

        b_update = np.zeros(shape=b_guess.shape)

        # Active set strategy
        if niter % active_set_cycle == 0:
            range_j = safe_range
        else:
            range_j = active_range(b_guess)

        for j in range_j:

            # Extract j^{th} covariate vector
            Xj = X[:, j].reshape(-1, 1)

            # Compute OLS solution for β_j taking β_{-j} as given
            bj = b_guess[j] + Xj.T @ (y - X @ b_guess)/N

            # Update guess for j^{th} coordinate using LASSO closed form solution under CDG
            b_update[j] = dual_sol(bj, lmbda)

        b0 = y_mean - np.dot(X_mean, b_update)
        dist = norm(b_update - b_guess, ord=np.inf)

        output["estimate"].append(np.append(b0, b_update))
        output["objectives"].append(lasso_obj(b_update))
        output["steps"].append(dist)

        niter = niter + 1
        b_guess = b_update

        if dist <= eps:
            output["status"] = "convergence"

    output["status"] = "max_iter exceed"

    return output