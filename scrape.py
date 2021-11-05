import pandas as pd
import numpy as np
import opendatasets as od

# This is how to get the datasets 
# # # #  # # # # # #  # # # # # #  # # # # # # # #  # # # # # # #  # # # # #
# pip3 install openddatasets for mac                                       #
# {"username":"anthonysutherland","key":"609037051118a8e60667c79f2159f312"}#
# # # # #  # # # # # #  # # # # # # #  # # # # # # # #  # # # # # # #  # # # 

# uncomment line below to get new set of data 
#od.download("https://www.kaggle.com/paultimothymooney/stock-market-data/download")

path_forbes = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_nasdaq = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_nyse = 'stock-market-data/stock_market_data/forbes2000/csv/'
path_sp500 = 'stock-market-data/stock_market_data/forbes2000/csv/'

filename = (path_nasdaq + "AMZN.csv")
df = pd.read_csv(filename)
print(df)