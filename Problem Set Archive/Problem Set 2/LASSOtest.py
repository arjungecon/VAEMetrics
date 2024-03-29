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
from LASSOHelperAGTT import lasso_wrapper_sequential, lambda_zero, lasso_cdg, lasso_wrapper_parallel
import cProfile

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amssymb}',
    r'\usepackage{amsmath}',
    r'\usepackage{xcolor}',
    r'\renewcommand*\familydefault{\sfdefault}']
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['pgf.preamble']  = [
    r'\usepackage[utf8x]{inputenc}',
    r'\usepackage{amssymb}',
    r'\usepackage[T1]{fontenc}',
    r'\usepackage{amsmath}',
    r'\usepackage{sansmath}']

inv, ax, norm = np.linalg.inv, np.newaxis, np.linalg.norm
randint = np.random.randint

N_obs = 200
N_param = 120

# Simulate data used in exercise
X, u, b = np.random.randn(N_obs, N_param), np.random.randn(N_obs, 1), np.random.randn(N_param, 1)

# Set intercept
X[:, 0] = 1.

# Random number of coefficients set to zero
n_0 = 70
b[randint(1, N_param, n_0), :] = 0

# Set outcome variable
Y = X @ b + u

lasso_est = lasso_wrapper_parallel(b*0, y=Y, X=X, standardized=False, num_lambda=100, min_factor=0.0005, k_fold=True)

# print(lasso_est['status'])

# lmbda, cv = lasso_K_fold(b_start=0*b, y=Y, X=X)

# @njit(parallel=True)
# Def fun(x,c):
# for k in prange(k)

lasso_res = np.array(lasso_est[0])
lasso_res = lasso_res[lasso_res[:, 1] < 1e10, :]
lasso_res[:, 1] = lasso_res[:, 1]/lasso_res[:, 1].max()

fig1, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True, dpi=400)

ax1.plot(lasso_res[:, 0], lasso_res[:, 1])
ax1.set_xlabel(r'$\lambda$', fontsize=13)
plt.show()
