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


class Tree:

    def __init__(self):

        """
        Constructor for this class. Creates the attributes corresponding to each node in the tree.
        """

        self.Y = None
        self.index = None
        self.left = None
        self.right = None

    def createTerminalNode(self, data, index):

        """
        Creates the terminal node for the tree.
        :param data: Dataset that contains the 1-D variables X and Y
        :param index: Observation indices that belongs to the terminal node
        """

        self.index = index
        self.Y = data.iloc[index]['Y'].mean()

    def addLeftNode(self, data, index):
        """
        Creates the left node for the tree.
        :param data: Dataset that contains the 1-D variables X and Y
        :param index: Observation indices that belongs to the left node
        """

        self.left = Tree()
        self.left.createTerminalNode(data, index)

    def addRightNode(self, data, index):
        """
        Creates the right node for the tree.
        :param data: Dataset that contains the 1-D variables X and Y
        :param index: Observation indices that belongs to the right node
        """

        self.right = Tree()
        self.right.createTerminalNode(data, index)

    @staticmethod
    def objectiveMSE(data, value):

        """
        Computes the objective function used to partition the tree node for a given partition value.
        :param data: Dataset that contains the 1-D variables X and Y.
        :param value: Partitioning value.
        """

        y_left, y_right = data.query('X <= @value')['Y'].mean(), data.query('X > @value')['Y'].mean()

        return np.sum((data.query('X <= @value')['Y'] - y_left)**2) + \
            np.sum((data.query('X > @value')['Y'] - y_right)**2)

    def growTree(self, data):

        """
        Grows the tree depending on the specification of the terminal node. It either preserves the
        terminal node as is if the input/terminal node is invalid, or it updates the tree by partitioning
        the indices into left and right leaves using a rectangular partitioning rule.

        :param data: Dataset that contains the 1-D variables X and Y.
        """

        if ~(self.index is None or len(self.index) == 1):

            data_node = data.iloc[self.index]

            vec_x = np.linspace(start=data_node['X'].min(), stop=data_node['X'].max(), num=100)

            mse_val = np.vectorize(self.objectiveMSE, excluded=['data'])(data=data_node, value=vec_x)
            c_node = vec_x[np.argmin(mse_val)]

            left_index = np.array(data_node.query('X <= @c_node').index)
            right_index = np.setdiff1d(self.index, left_index)

            self.addLeftNode(data=data, index=left_index)
            self.addRightNode(data=data, index=right_index)

    def predictTree(self, index):
        """
            Predicts the mean outcome in the terminal node for a given observation index.

            :param index: Observation index.
        """

        terminal_node = self

        while not(terminal_node.left is None and terminal_node.right is None):

            if index in terminal_node.left.index:

                terminal_node = terminal_node.left

            elif index in terminal_node.right.index:

                terminal_node = terminal_node.right

            else:

                print('Error - Cannot find index.')

                return None

        return terminal_node.Y
