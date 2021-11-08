import pandas as pd
import numpy as np
import opendatasets as od
import argparse

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
parser.add_argument('--stock'   , default='AMZN', help='stock symbol to predict')
parser.add_argument('--showfigs', default=False , help='show plots while running' , action='store_true')
parser.add_argument('--getdata' , default=False , help='download data from Kaggle', action='store_true')
args = parser.parse_args()

if args.getdata:
    od.download("https://www.kaggle.com/paultimothymooney/stock-market-data/download")

path_forbes = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_nasdaq = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_nyse = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_sp500 = 'stock-market-data/stock_market_data/forbes2000/csv/'


################### Read Data File ########################################

filename = (path_nasdaq + args.stock+'.csv')
df = pd.read_csv(filename)

################### Data Pre-processing ###################################

print(df.head())
print(df.describe())

# Look at correlation of features
sbn.heatmap(df.corr())
plt.title('Heatmap of A')
plt.savefig('data_heatmap.png')
plt.show() if args.showfigs else plt.clf()

# Visualize all features plotted against each other
sbn.pairplot(df)
plt.savefig('data_pairplot.png')
plt.show() if args.showfigs else plt.clf()

# Visualize the adjusted closing price over time
plt.figure(figsize=(20,10))
plt.plot(range(df.shape[0]), df['Adjusted Close'])
plt.xticks(range(0, df.shape[0],500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adjusted Closing Price', fontsize=18)
plt.savefig('historic_closeprice_plot.png')
plt.show() if args.showfigs else plt.clf()

# Move data to numpy arrays and split into training and testing sets
df['Date'] = pd.to_datetime(df['Date'])
v = df.to_numpy()
X = v[:,0:6] # create X data excluding date and adjusted closing price
y = v[:,6]   # create y data using adjusted closing price -> target
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


################## Prediction Techniques ##################################

### TODO: Implement an sklearn method to predict future prices
###       We can test linear regression against some other method(s)
###       Possible other methods: Gaussian, CNN, Time Series Modeling
###                               all are available in sklearn
###
### Links: https://www.ethanrosenthal.com/2019/02/18/time-series-for-scikit-learn-people-part3/
###        https://www.datacamp.com/community/tutorials/lstm-python-stock-market
###        https://www.kaggle.com/razvangeorgecostea/practice-machine-learning-on-apple-stock
###        
