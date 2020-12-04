"""
Final Project
Will Suratt
CS 021

Data Processor
==============

This program will process the raw data downloaded from Cboe into
data that is useful to train the model and output this data as
a .csv file.

*** This program takes about 2 hours to fully run. ***

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import alpaca_trade_api as tradeapi
import time
import math
import datetime
from datetime import timedelta, date
import os

def main():
    """

    Begins program execution: Connects to api, then reads in raw data to a 
    DataFrame. This data is then passed into the process_data function to
    get identify and calculate data that is relevant for training the model.

    """
    # Connect to Alpaca API
    base_url = 'https://paper-api.alpaca.markets'
    api_key_id = 'PKTR1CFQXFSDYX3SXXSC'
    api_secret = 'RFAcxcowzJgNlvPuFqp3bzqoIyRuCmC4C2s78njO'

    api = tradeapi.REST(
        base_url=base_url,
        key_id=api_key_id,
        secret_key=api_secret
    )

    # Get raw data
    cwd = os.getcwd()
    input_path = cwd + '/data/'
    raw_data = pd.read_csv(input_path + 'options_data.csv')

    # Process data
    process_data(api, raw_data)

def process_data(api, raw_data):
    """

    Two parameter variables, api (connects to Alpaca API) and raw_data (DataFrame).
    Processes data by creating new DataFrame and populating it with relevant data
    for training the model. To do this, it gets price data from before the current 
    raw_data date to calculate historical volatility. Then calculate break even up 
    and down percentage price movements from raw_data. Finally gets output whether
    position was profitable or not using price data from current raw_data date
    until options' expiration date.

    """
    # Create empty DataFrame to save processed data
    columns = ['Symbol', 'ExpirationDate', 'CallPremium', 'PutPremium', 'Volatility', 'BEUp', 'BEDown', 'Output']
    processed_data = pd.DataFrame(columns=columns)

    # Loop through each row of data
    for i in range(len(raw_data.index)):
        try:
            # 
            if round(raw_data['UnderlyingPrice'][i]) == raw_data['Strike'][i] and raw_data['Type'][i] == 'call':
                
                # Get current date for option and process it
                current_date = datetime.datetime.strptime(raw_data[' DataDate'][i], '%m/%d/%Y').strftime('%Y-%m-%d')
                current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d')
                current_date = datetime.date(current_date.year, current_date.month, current_date.day)

                # Get option expiration data and process it
                expiration_date = datetime.datetime.strptime(raw_data['Expiration'][i], '%m/%d/%Y').strftime('%Y-%m-%d')
                expiration_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                expiration_date = datetime.date(expiration_date.year, expiration_date.month, expiration_date.day)
                # Get number of days between current date and expiration date
                delta = expiration_date - current_date
                days_expiration = delta.days

                # Get date number of days between current date and expiration date before current date
                previous_date = (current_date-timedelta(days=days_expiration)).isoformat()

                # Pause to keep IP from getting blocked
                time.sleep(1)
                # Get close price data for specific stock from previouse_date to current_date
                stock_history_train = api.get_aggs(symbol=raw_data['UnderlyingSymbol'][i], timespan='day', multiplier=1, _from=previous_date, to=str(current_date)).df
                close_data = stock_history_train["close"]

                # Double check there is close data for the specific stock before continuing
                if close_data.size:

                    # Calculate mean
                    mean = close_data.mean()

                    # Calculate standard deviation and normalize it by converting it to a percentage
                    total = 0
                    for value in close_data:
                        deviation = value - mean
                        total += deviation**2

                    volatility = total / close_data.size
                    std_dev = math.sqrt(volatility)
                    percentage = std_dev / close_data[-1]
                    
                    # Get call premium and put premium by averaging Ask and Bid
                    call_premium = (raw_data['Bid'][i] + raw_data['Ask'][i]) / 2
                    put_premium = (raw_data['Bid'][i+1] + raw_data['Ask'][i+1]) / 2

                    # Calculate price change up or down to break even
                    be_up = (raw_data['Strike'][i] - raw_data['UnderlyingPrice'][i]) + (call_premium + put_premium)
                    be_down = abs((raw_data['Strike'][i+1] - raw_data['UnderlyingPrice'][i+1]) - (put_premium + call_premium))

                    # Change this price change into percentage change
                    be_up_percent = be_up / raw_data['UnderlyingPrice'][i]
                    be_down_percent = be_down / raw_data['UnderlyingPrice'][i+1]

                    # Set be_up and be_down to break even prices
                    be_up = raw_data['UnderlyingPrice'][i] + be_up
                    be_down = raw_data['UnderlyingPrice'][i+1] - be_down

                    # Get close price data for specific stock from current_date to expiration_date
                    stock_history_train = api.get_aggs(symbol=raw_data['UnderlyingSymbol'][i], timespan='day', multiplier=1, _from=str(current_date), to=str(expiration_date)).df
                    close_data = stock_history_train["close"]

                    # Figure out if position was profitable
                    output = 0
                    for value in close_data:
                        if value >= be_up or value <= be_down:
                            output = 1

                    # Append row to Dataframe
                    new_row = {'Symbol': raw_data['UnderlyingSymbol'][i], 'ExpirationDate': expiration_date, 'CallPremium': call_premium, 'PutPremium': put_premium, 'Volatility': percentage, 'BEUp': be_up_percent, 'BEDown': be_down_percent, 'Output': output}
                    processed_data = processed_data.append(new_row, ignore_index=True)
                    print('Row appended: ', i)
        except ValueError:
            print('Invalid value. Row skipped.')

    # Set current directory
    cwd = os.getcwd()
    output_path = cwd + '/data/'
    processed_data.to_csv(output_path + 'processed_data.csv', index=False)
    print('File saved.')
# Run program
main()