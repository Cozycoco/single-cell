import os, gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

data_loc = "/Users/annabellahe/github/single_cell/data"

#Load data
cite_train_inputs = os.path.join(data_loc, "test_cite_inputs.h5")
cite_train_targets = os.path.join(data_loc, "train_cite_targets.h5")
cite_test_inputs = os.path.join(data_loc, "train_cite_inputs.h5")

#EDA
df_cite_train_x = pd.read_hdf(cite_train_inputs)
display(df_cite_train_x.head())
print('Shapes:', df_cite_train_x.shape, df_cite_test_x.shape, df_cell_cite.shape)
print("Missing values:", df_cite_train_x.isna().sum().sum(), df_cite_test_x.isna().sum().sum())
print("Genes which never occur in train:", (df_cite_train_x == 0).all(axis=0).sum())
print("Genes which never occur in test: ", (df_cite_test_x == 0).all(axis=0).sum())
print(f"Zero entries in train: {(df_cite_train_x == 0).sum().sum() / df_cite_train_x.size:.0%}")
print(f"Zero entries in test:  {(df_cite_test_x == 0).sum().sum() / df_cite_test_x.size:.0%}")

df_cite_test_x = None # release the memory
