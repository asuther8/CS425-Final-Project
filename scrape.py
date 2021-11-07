import pandas as pd
import numpy as np
import opendatasets as od
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sbn
sbn.set()

# This is how to get the datasets 
# # # #  # # # # # #  # # # # # #  # # # # # # # #  # # # # # # #  # # # # #
# pip3 install openddatasets for mac                                       #
# {"username":"anthonysutherland","key":"609037051118a8e60667c79f2159f312"}#
# # # # #  # # # # # #  # # # # # # #  # # # # # # # #  # # # # # # #  # # # 

parser = argparse.ArgumentParser(description="Script for estimating stock prices")
parser.add_argument('--showfigs', default=False, help='show plots while running' , action='store_true')
parser.add_argument('--getdata' , default=False, help='download data from Kaggle', action='store_true')
args = parser.parse_args()

if args.getdata:
    od.download("https://www.kaggle.com/paultimothymooney/stock-market-data/download")

path_forbes = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_nasdaq = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_nyse = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_sp500 = 'stock-market-data/stock_market_data/forbes2000/csv/'

################### Read Data File ########################################

filename = (path_nasdaq + "AMZN.csv")
df = pd.read_csv(filename)

################### Data Pre-processing ###################################

print(df)
print(df.describe())

sbn.heatmap(df.corr())
plt.savefig('data_heatmap.png')
plt.show() if args.showfigs else plt.clf()

sbn.pairplot(df)
plt.savefig('data_pairplot.png')
plt.show() if args.showfigs else plt.clf()

v = df.to_numpy()
X = v[:,1:6] # create X data excluding date and adjusted closing price
y = v[:,6]   # create y data using adjusted closing price -> target

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

################## Gaussian Regression ###################################

### TODO: This does not work yet! ###
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X_train, y_train)
gpr.score(X_train, y_train)
gpr.predict(X_test, return_std=True)
#####################################

