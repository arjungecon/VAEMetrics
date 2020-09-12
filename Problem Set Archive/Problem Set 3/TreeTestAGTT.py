from TreeHelperAGTT import Tree

# Standard Python Imports

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
import cProfile
import warnings

warnings.filterwarnings("ignore")

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

np.random.seed(101)
N_obs = 100

data = dict()
data['X'] = np.random.normal(size=N_obs)
data['Y'] = data['X'] ** 2 + np.random.normal(size=N_obs)
data = pd.DataFrame(data)

tree_ex = Tree()
tree_ex.createTerminalNode(data=data, index=data.index)
tree_ex.growTree(data=data)
tree_ex.left.growTree(data=data)
tree_ex.right.growTree(data=data)
tree_ex.left.left.growTree(data=data)
tree_ex.left.right.growTree(data=data)
tree_ex.right.left.growTree(data=data)
tree_ex.right.right.growTree(data=data)
