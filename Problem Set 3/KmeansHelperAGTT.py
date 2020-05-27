{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "\n",
    "os.environ['PYTHONHOME'] = r\"C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python37\"\n",
    "os.environ['PYTHONPATH'] = r\"C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\"\n",
    "os.environ['R_HOME'] = r\"C:\\Program Files\\R\\R-3.6.3\"\n",
    "os.environ['R_USER'] = r\"C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\rpy2\"\n",
    "\n",
    "import rpy2\n",
    "import rpy2.robjects as robjects\n",
    "\n",
    "pi = robjects.r['pi']\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.patches as mpatches\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "from matplotlib import rc\n",
    "import statsmodels.api as sm\n",
    "from scipy.stats import zscore\n",
    "import scipy as sp\n",
    "from numpy import random, linalg\n",
    "from scipy import sparse, stats\n",
    "import itertools as it\n",
    "from sklearn.preprocessing import StandardScaler as scaler\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.model_selection import KFold\n",
    "from joblib import Parallel, delayed\n",
    "import multiprocessing\n",
    "import rpy2.robjects as robjects\n",
    "\n",
    "\n",
    "matplotlib.rcParams['text.usetex'] = True\n",
    "matplotlib.rcParams['text.latex.preamble'] = [\n",
    "    r'\\usepackage{amssymb}',\n",
    "    r'\\usepackage{amsmath}',\n",
    "    r'\\usepackage{xcolor}',\n",
    "    r'\\renewcommand*\\familydefault{\\sfdefault}']\n",
    "matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'\n",
    "matplotlib.rcParams['pgf.preamble'] = [\n",
    "    r'\\usepackage[utf8x]{inputenc}',\n",
    "    r'\\usepackage{amssymb}',\n",
    "    r'\\usepackage[T1]{fontenc}',\n",
    "    r'\\usepackage{amsmath}',\n",
    "    r'\\usepackage{sansmath}']\n",
    "\n",
    "inv, ax, norm = np.linalg.inv, np.newaxis, np.linalg.norm\n",
    "randint = np.random.randint\n",
    "\n",
    "\n",
    "def kmeans_objective(K, centers, groups, X):\n",
    "\n",
    "    return np.sum(norm(X-centers[groups.astype(int),],ord=2)**2)\n",
    "\n",
    "def kmeans_updateCenters(K, groups, X):\n",
    "    # read dimensions\n",
    "    N = X.shape[0]\n",
    "    p = X.shape[1]\n",
    "    centers=np.empty(shape=(K,p))\n",
    "    for i in range(K):\n",
    "        centers[i,] = np.mean(X[np.where(groups==i)[0],],axis=0)\n",
    "    return centers\n",
    "\n",
    "def kmeans_updateGroups(K, centers, X): \n",
    "    # read dimensions\n",
    "    N = X.shape[0]\n",
    "    p = X.shape[1]\n",
    "    groups=np.empty(shape=(N,))\n",
    "    for i in range(N):\n",
    "        groups[i,] = np.argmin(np.apply_along_axis(lambda row:np.linalg.norm(row,ord=2)**2,1,np.tile(X[i],(K,1))-centers))\n",
    "    return groups\n",
    "\n",
    "def kmeans_estimateGroups(K, initCenters, X, tol=1e-06, maxiter=1000):\n",
    "    keyDict = {\"groups\",\"centers\",\"objectives\",\"steps\",\"status\"}\n",
    "    output = dict([(key, []) for key in keyDict])\n",
    "    centers = initCenters\n",
    "    for i in range(maxiter):\n",
    "        groups = kmeans_updateGroups(K, centers, X)\n",
    "        new_centers = kmeans_updateCenters(K, groups, X)\n",
    "        diff = np.max(np.apply_along_axis(lambda row:np.linalg.norm(row,ord=2)**2,1,new_centers-centers))\n",
    "        if np.isnan(np.min(new_centers))==True:\n",
    "            output['groups'] = groups\n",
    "            output['centers'] = new_centers\n",
    "            output['objectives'] = np.append(output['objectives'],kmeans_objective(K, new_centers, groups, X))\n",
    "            output['steps'] = np.append(output['steps'],diff)\n",
    "            output['status'] = \"empty group appeared.\"\n",
    "            return output\n",
    "        if diff < tol:\n",
    "            output['groups'] = groups\n",
    "            output['centers'] = new_centers\n",
    "            output['objectives'] = np.append(output['objectives'],kmeans_objective(K, new_centers, groups, X))\n",
    "            output['steps'] = np.append(output['steps'],diff)\n",
    "            output['status'] = \"algorithm converged.\"\n",
    "            return output\n",
    "        output['groups'] = groups\n",
    "        output['centers'] = new_centers\n",
    "        output['objectives'] = np.append(output['objectives'],kmeans_objective(K, new_centers, groups, X))\n",
    "        output['steps'] = np.append(output['steps'],diff)    \n",
    "        centers = new_centers\n",
    "    output['status'] = \"max iteration reached.\"\n",
    "    return output\n",
    "\n",
    "def generateSample(N, p, K, meanScale=1, varScale=1):\n",
    "    # generate random mean vectors\n",
    "    means =  meanScale * np.random.normal(size=(K,p))\n",
    "    \n",
    "    # generate random variances\n",
    "    var = np.zeros((K,p,p))\n",
    "    for k in range(K):\n",
    "        varChol = np.zeros((p,p))\n",
    "        varChol[np.tril_indices(p)] = varScale * np.random.normal(size=int(p*(p+1)/2)) \n",
    "        var[k,:,:]=varChol\n",
    " \n",
    "    # generate standard normal draws, into p * N matrix\n",
    "    draws = np.random.normal(size=(p,N))\n",
    "    # generate group membershipsp\n",
    "    groups = np.random.randint(low=0, high=K, size=N, dtype=int)\n",
    "    # location and scale shift according to the mixture components\n",
    "    for j in range(K):\n",
    "        draws[:,groups == j] = np.transpose(means[j,:])[:, np.newaxis] + var[j,:,:] @ draws[:,groups == j]\n",
    "\n",
    "    # return draws and the true group membership\n",
    "    return {'draws':np.transpose(draws),'centers':means,'groups':groups,'var':var}"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}