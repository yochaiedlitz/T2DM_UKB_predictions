import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib import rcParams as rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib.gridspec as gridspec

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc

import seaborn as sns

from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

from textwrap import wrap
import shap
from collections import OrderedDict
import random
import re

# from . import UKBB_Func
def standarise_df(df):
    fit_col = df.columns
    x_std_col = [x for x in fit_col if not x.endswith("_na")]
    x_na_col = [x for x in fit_col if x.endswith("_na")]
    x_train_std = df[x_std_col]
    x_train_std = (x_train_std - np.mean(x_train_std, axis=0)) / np.std(x_train_std, axis=0)
    x_train_std_na_col = x_train_std.loc[:, x_train_std.isna().sum() > 0].columns.values
    x_train_std.loc[:, x_train_std.isna().sum() > 0] = df.loc[:, x_train_std_na_col]
    x_train_std[x_na_col] = df[x_na_col]
    return x_train_std