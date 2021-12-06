from datetime import timedelta
import pandas as pd
import numpy as np
import opendatasets as od
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

# Constants
RSEED = 425
TEST_PORTION = 0.10

# Preprocess data
def preprocess_data(df):

    # Add column to dataframe converting datetime string to datetime
    df['Date_Datetime'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Add column to dataframe converting datetime string to epoch seconds
    df['Date_Int'] = df['Date_Datetime'].view(int) / 10**9

    # Only use the date as a feature for these runs
    X = df['Date_Int'].to_numpy()
    y = df['Adjusted Close'].to_numpy()

    # Split stock history data into training and test data to determine how well trained model predicts historical data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PORTION, random_state=RSEED)

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Scale training and test dates (in epochs) based on training dates
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Find scaled X date feature for every day from stock beginning to future to predict/graph whole space
    X_every_day_dates = pd.date_range(df.iloc[0]['Date_Datetime'], df.iloc[-1]['Date_Datetime'] + pd.DateOffset(years=10))
    X_every_day_ints = X_every_day_dates.view(int) / 10**9
    X_every_day_scaled = scaler.transform(X_every_day_ints.reshape(-1,1))

    return X_train_scaled, X_test_scaled, y_train, y_test, X_every_day_dates, X_every_day_scaled


# Train regression model via artificial neural network
def nn_regression(stock, df, X_train, X_test, y_train, y_test, X_every_day_dates, X_every_day):

    # Train/fit model
    mlp_regression = MLPRegressor(random_state=RSEED, verbose=True, max_iter=10000, alpha=0)
    mlp_regression.fit(X_train, y_train)
    
    # Make predictions on test data to report performance of model on historical stock data
    y_pred_test = mlp_regression.predict(X_test)
    print(f'NN R^2 Score On Historical Data: {metrics.r2_score(y_test, y_pred_test)}')

    # Make predictions on every day from beginning of stock to future
    y_pred_every_day = mlp_regression.predict(X_every_day)

    # Plot true adjusted close price history for stock and predicted using model
    years = mdates.YearLocator(base=2)   # every other year
    years_fmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots()
    plt.plot_date(df['Date_Datetime'], df['Adjusted Close'], label = "Actual Adjusted Close", linestyle = "-", marker=None)
    plt.plot_date(X_every_day_dates, y_pred_every_day, label = "Predicted Adjusted Close", linestyle = "-", marker=None)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adjusted Closing Price', fontsize=18)
    plt.title(stock + ": MLP Regression Predicted Adjusted Closing Price vs. Actual")
    plt.legend()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    fig.autofmt_xdate()
    plt.savefig('ann_date_est_plot_' + stock + '.png')
    plt.clf()

    return mlp_regression



def main():

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Script for estimating stock prices")
    parser.add_argument('--stock'   , default='AMZN', help='stock symbol to predict')
    parser.add_argument('--showfigs', default=False , help='show plots while running' , action='store_true')
    parser.add_argument('--getdata' , default=False , help='download data from Kaggle', action='store_true')
    args = parser.parse_args()

    if args.getdata:
        od.download("https://www.kaggle.com/paultimothymooney/stock-market-data/download")

    path_nasdaq = 'stock-market-data/stock_market_data/forbes2000/csv/'

    # Read data file
    filename = (path_nasdaq + args.stock + '.csv')
    df = pd.read_csv(filename)

    # Process data to get training and test data
    X_train, X_test, y_train, y_test, X_every_day_dates, X_every_day = preprocess_data(df)

    # Train ANN for regression on historical stock data
    mlp_regression = nn_regression(args.stock, df, X_train, X_test, y_train, y_test, X_every_day_dates, X_every_day)

    

if __name__ == "__main__":
    main()