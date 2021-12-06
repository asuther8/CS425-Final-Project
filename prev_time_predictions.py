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

    print(df)

    # Add dataframe columns for previous week's average feature values
    week_start_datetime = df.iloc[0]['Date_Datetime']
    num_days = 0
    week_low_sum = 0
    week_open_sum = 0
    week_volume_sum = 0
    week_high_sum = 0
    week_close_sum = 0
    prev_week_low_mean = -1
    prev_week_open_mean = -1
    prev_week_volume_mean = -1
    prev_week_high_mean = -1
    prev_week_close_mean = -1

    for index, date in df['Date_Datetime'].items():
        if (date - week_start_datetime).days >= 7:
            if num_days > 0:
                prev_week_low_mean = week_low_sum / num_days
                prev_week_open_mean = week_open_sum / num_days
                prev_week_volume_mean = week_volume_sum / num_days
                prev_week_high_mean = week_high_sum / num_days
                prev_week_close_mean = week_close_sum / num_days

            week_start_datetime = date
            num_days = 0
            week_low_sum = 0
            week_open_sum = 0
            week_volume_sum = 0
            week_high_sum = 0
            week_close_sum = 0
        
        df.at[index, 'Low_Prev_Week_Mean'] = prev_week_low_mean
        df.at[index, 'Open_Prev_Week_Mean'] = prev_week_open_mean
        df.at[index, 'Volume_Prev_Week_Mean'] = prev_week_volume_mean
        df.at[index, 'High_Prev_Week_Mean'] = prev_week_high_mean
        df.at[index, 'Close_Prev_Week_Mean'] = prev_week_close_mean

        week_low_sum += df.iloc[index]['Low']
        week_open_sum += df.iloc[index]['Open']
        week_volume_sum += df.iloc[index]['Volume']
        week_high_sum += df.iloc[index]['High']
        week_close_sum += df.iloc[index]['Close']

        num_days = num_days + 1

    print(df.head(15))
    print(df.tail(25))

    features = ['Date_Int','Low_Prev_Week_Mean','Open_Prev_Week_Mean','Volume_Prev_Week_Mean','High_Prev_Week_Mean','Close_Prev_Week_Mean']

    # Only use the date as a feature for these runs
    X = df[features].to_numpy()
    y = df['Adjusted Close'].to_numpy()

    X_dates_unfiltered = df['Date_Datetime'].to_numpy()
    X_dates = X_dates_unfiltered[5:]

    # Filter out data points without previous week info
    X_filtered = X[5:]
    y_filtered = y[5:]
    X_future = X[6148:]
    y_future = y[6148:]

    # Split stock history data into training and test data to determine how well trained model predicts historical data
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=TEST_PORTION, random_state=RSEED)

    # Scale training and test dates (in epochs) based on training dates
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_future_scaled = scaler.transform(X_future)
    X_scaled = scaler.transform(X[5:])

    return X_train_scaled, X_test_scaled, y_train, y_test, X_future_scaled, y_future, X_scaled, X_dates


# Train regression model via artificial neural network
def nn_regression(stock, df, X_train, X_test, y_train, y_test, X_future, y_future, X, X_dates):

    # Train/fit model
    mlp_regression = MLPRegressor(random_state=RSEED, verbose=True, max_iter=10000, alpha=0)
    mlp_regression.fit(X_train, y_train)
    
    # Make predictions on test data to report performance of model on historical stock data
    y_pred_test = mlp_regression.predict(X_test)
    print(f'NN R^2 Score On Historical Data: {metrics.r2_score(y_test, y_pred_test)}')

    # Make predictions on every day from beginning of stock to future
    y_pred_future = mlp_regression.predict(X_future)
    print(f'NN R^2 Score On Last Week of "Future" Data: {metrics.r2_score(y_future, y_pred_future)}')

    y_pred_everything = mlp_regression.predict(X)

    # Plot true adjusted close price history for stock and predicted using model
    years = mdates.YearLocator(base=2)   # every other year
    years_fmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots()
    plt.plot_date(df['Date_Datetime'], df['Adjusted Close'], label = "Actual Adjusted Close", linestyle = "-", marker=None)
    plt.plot_date(X_dates, y_pred_everything, label = "Predicted Adjusted Close", linestyle = "-", marker=None)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adjusted Closing Price', fontsize=18)
    plt.title(stock + ": MLP Regression Predicted Adjusted Closing Price vs. Actual")
    plt.legend()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    fig.autofmt_xdate()
    plt.show()
    plt.savefig('ann_prev_time_est_plot_' + stock + '.png')
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
    X_train, X_test, y_train, y_test, X_future, y_future, X, X_dates = preprocess_data(df)

    # Train ANN for regression on historical stock data
    mlp_regression = nn_regression(args.stock, df, X_train, X_test, y_train, y_test, X_future, y_future, X, X_dates)

    

if __name__ == "__main__":
    main()