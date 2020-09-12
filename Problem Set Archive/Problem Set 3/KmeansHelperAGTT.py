import os


os.environ['PYTHONHOME'] = r"C:\Users\me\AppData\Local\Programs\Python\Python37"
os.environ['PYTHONPATH'] = r"C:\Users\me\AppData\Local\Programs\Python\Python37\Lib\site-packages"
os.environ['R_HOME'] = r"C:\Program Files\R\R-3.6.3"
os.environ['R_USER'] = r"C:\Users\me\AppData\Local\Programs\Python\Python37\Lib\site-packages\rpy2"

import rpy2
import rpy2.robjects as robjects

pi = robjects.r['pi']

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import statsmodels.api as sm
from scipy.stats import zscore
import scipy as sp
from numpy import random, linalg
from scipy import sparse, stats
import itertools as it
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
import rpy2.robjects as robjects


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


def kmeans_objective(K, centers, groups, X):

    return np.sum(norm(X-centers[groups.astype(int),],ord=2)**2)

def kmeans_updateCenters(K, groups, X):
    # read dimensions
    N = X.shape[0]
    p = X.shape[1]
    centers=np.empty(shape=(K,p))
    for i in range(K):
        centers[i,] = np.mean(X[np.where(groups==i)[0],],axis=0)
    return centers

def kmeans_updateGroups(K, centers, X): 
    # read dimensions
    N = X.shape[0]
    p = X.shape[1]
    groups=np.empty(shape=(N,))
    for i in range(N):
        groups[i,] = np.argmin(np.apply_along_axis(lambda row:np.linalg.norm(row,ord=2)**2,1,np.tile(X[i],(K,1))-centers))
    return groups

def kmeans_estimateGroups(K, initCenters, X, tol=1e-06, maxiter=1000):
    keyDict = {"groups","centers","objectives","steps","status"}
    output = dict([(key, []) for key in keyDict])
    centers = initCenters
    for i in range(maxiter):
        groups = kmeans_updateGroups(K, centers, X)
        new_centers = kmeans_updateCenters(K, groups, X)
        diff = np.max(np.apply_along_axis(lambda row:np.linalg.norm(row,ord=2)**2,1,new_centers-centers))
        if np.isnan(np.min(new_centers))==True:
            output['groups'] = groups
            output['centers'] = new_centers
            output['objectives'] = np.append(output['objectives'],kmeans_objective(K, new_centers, groups, X))
            output['steps'] = np.append(output['steps'],diff)
            output['status'] = "empty group appeared."
            return output
        if diff < tol:
            output['groups'] = groups
            output['centers'] = new_centers
            output['objectives'] = np.append(output['objectives'],kmeans_objective(K, new_centers, groups, X))
            output['steps'] = np.append(output['steps'],diff)
            output['status'] = "algorithm converged."
            return output
        output['groups'] = groups
        output['centers'] = new_centers
        output['objectives'] = np.append(output['objectives'],kmeans_objective(K, new_centers, groups, X))
        output['steps'] = np.append(output['steps'],diff)    
        centers = new_centers
    output['status'] = "max iteration reached."
    return output

def generateSample(N, p, K, meanScale=1, varScale=1):
    # generate random mean vectors
    means =  meanScale * np.random.normal(size=(K,p))
    
    # generate random variances
    var = np.zeros((K,p,p))
    for k in range(K):
        varChol = np.zeros((p,p))
        varChol[np.tril_indices(p)] = varScale * np.random.normal(size=int(p*(p+1)/2)) 
        var[k,:,:]=varChol
 
    # generate standard normal draws, into p * N matrix
    draws = np.random.normal(size=(p,N))
    # generate group membershipsp
    groups = np.random.randint(low=0, high=K, size=N, dtype=int)
    # location and scale shift according to the mixture components
    for j in range(K):
        draws[:,groups == j] = np.transpose(means[j,:])[:, np.newaxis] + var[j,:,:] @ draws[:,groups == j]

    # return draws and the true group membership
    return {'draws':np.transpose(draws),'centers':means,'groups':groups,'var':var}