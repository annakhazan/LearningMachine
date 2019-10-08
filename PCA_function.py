import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def pca_data15(df):
	pca = PCA(n_components=15)
	pca.fit(df)
	return pd.DataFrame(pca.transform(df))

def pca_data100(df):
	pca = PCA(n_components=100)
	pca.fit(df)
	return pd.DataFrame(pca.transform(df))