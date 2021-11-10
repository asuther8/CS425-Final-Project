import pandas as pd
import pandas_ta
import numpy as np
import opendatasets as od
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sbn
sbn.set()

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

# Add ema (exponential moving average) to plot against close price
# This is a work-around for linear predictions bc time not a good x var
group_size = 10 # we can test with different values here for better fit?
df.ta.ema(close='Adjusted Close', length=group_size, append=True)
df = df.reindex(columns=(['EMA_10'] + list([a for a in df.columns if a != 'EMA_10']) ))
df.dropna(subset=["EMA_10"], inplace=True) # move to front -> easier splitting of data later
print(df.head()) # Verify that new EMA_10 col is at front with no NANs

# Move data to numpy arrays and split into training and testing sets
df['Date'] = pd.to_datetime(df['Date'])
v = df.to_numpy()
X = v[:,0:7] # create X data excluding date and adjusted closing price
y = v[:,7]   # create y data using adjusted closing price -> target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


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

# Linear Classifier using EMA instead of date
# This is used to predict the closing price of any day given that day's ema (calculated 
# using the closing values of the previous 10 days)
# Link for reference:
#     https://www.alpharithms.com/predicting-stock-prices-with-linear-regression-214618/

model = LinearRegression()
model.fit(X_train[:,0].reshape(-1,1), y_train)
y_pred = model.predict(X_test[:,0].reshape(-1,1))

print('Linear Abs. Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Linear R^2 Score: ', metrics.r2_score(y_test, y_pred))

# Visualize the linear model estimates against actual prices
vmin = int(min(min(y_test), min(y_pred)))
vmax = int(max(max(y_test), max(y_test)))
plt.figure(figsize=(20,10))
plt.scatter(y_test, y_pred, c='crimson', label='predicted')
plt.plot([vmin, vmax], [vmin,vmax], 'b-', label='optimal fit')
plt.xlabel('True Closing Price', fontsize=18)
plt.ylabel('Predicted Closing Price', fontsize=18)
plt.title("Predicted Price vs. Actual for Linear Regression")
plt.legend()
plt.savefig('closeprice_linear_est_plot.png')
plt.show() if args.showfigs else plt.clf()

plt.figure().clear()

# Similar to Linear Regression but using Quadratic Regression instead
quadmodel = make_pipeline(PolynomialFeatures(3), Ridge())
quadmodel.fit(X_train[:,0].reshape(-1,1), y_train)
qy_pred = quadmodel.predict(X_test[:,0].reshape(-1,1))

print('Quadratic Regression Abs. Error:', metrics.mean_absolute_error(y_test, qy_pred))
print('Quadratic R^2 Score: ', metrics.r2_score(y_test, qy_pred))

# Visualize the quadratic model estimates against actual prices
vmin = int(min(min(y_test), min(qy_pred)))
vmax = int(max(max(y_test), max(y_test)))
plt.figure(figsize=(20,10))
plt.scatter(y_test, qy_pred, c='crimson', label='predicted')
plt.plot([vmin, vmax], [vmin,vmax], 'b-', label='optimal fit')
plt.xlabel('True Closing Price', fontsize=18)
plt.ylabel('Predicted Closing Price', fontsize=18)
plt.title("Predicted Price vs. Actual for Quadratic Regression")
plt.legend()
plt.savefig('closeprice_quadratic_est_plot.png')
plt.show() if args.showfigs else plt.clf()