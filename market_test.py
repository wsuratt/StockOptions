"""
Final Project
Will Suratt
CS 021

Market Test
===========

This program will test the performance of using the previously trained model to 
predict which stocks from the S&P 500 index to open a straddle position on. 
This will be tested against using random predictions, opening all available 
straddle positions, and the S&P 500 index. These results will then be
displayed with a total investment balance printed and a graph of performance
over time for each investment strategy used.

*** This program takes about 30 minutes to fully run. ***

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import alpaca_trade_api as tradeapi
import time
import math
import random
import datetime
from datetime import timedelta, date
from keras.models import load_model
import os

INITIAL_INVESTMENT = 25000
# Options are generally sold per 100 contracts
NUM_CONTRACTS = 100
# Number of sets of random predictions to average
NUM_RANDOM = 5
START_DATE = '2019-08-15'
END_DATE = '2020-01-17'
CWD = os.getcwd()
DATA_PATH = CWD + '/data/'
MODEL_PATH = CWD + '/model/'
MODEL = load_model(MODEL_PATH + 'stock_model.h5')

def main():
    """

    Begins program execution: Connects to api, then gets the data used to train the model
    and creates a list of symbols of S&P 500 stocks. Calls process_data function and uses
    returned data to get model performance (model_results), random performance (random_results),
    buy all performance (buyall_results), and market performance (market_performance). Calls
    display_results function to display results of the various investment strategies' performance.

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

    positions_data = pd.read_csv(DATA_PATH + 'processed_data.csv')

    # Create list of S&P 500 symbols
    symbols = []
    my_file = open(DATA_PATH + "s&p500.txt", "r")
    content = my_file.read()
    symbols = content.split("\n")
    my_file.close()

    # Get number of days between START_DATE and END_DATE
    days = num_days(api)

    # Get relevant data for testing
    positions = process_data(api, positions_data, symbols)

    # Get model performance
    model_final, model_performance = model_results(api, days, positions, symbols)

    # Get random performance
    random_final, random_performance = random_results(api, days, positions, symbols)

    # Get buy all available positions performance
    buyall_final, buyall_performance = buyall_results(api, days, positions, symbols)

    # Get market performance
    market_final, market_performance = market_results(api, days)

    # Display results
    display_results(model_final, random_final, buyall_final, market_final, 
                    model_performance, random_performance, buyall_performance, market_performance)



def process_data(api, positions_data, symbols):
    """

    Three parameter variables, api (connects to Alpaca API), positions data (DataFrame), 
    and symbols (list). Adds model prediction and random prediction to the positions data DataFrame
    for each straddle position. Returns updated DataFrame.

    """
    # Create empty DataFrame to copy S&P 500 symbols into
    columns = ['Symbol', 'ExpirationDate', 'CallPremium', 'PutPremium', 'Volatility', 'BEUp', 'BEDown', 'Days', 'Output']
    positions = pd.DataFrame(columns=columns)

    # Remove symbols and their data that aren't in S&P 500
    for symbol in symbols:
        positions = positions.append(positions_data.where(positions_data['Symbol'] == symbol).dropna().reset_index(drop = True))

    # Create empty Series to collect model predictions
    model_predictions = pd.Series()  

    # Add model predictions to Series
    for i in range(len(positions.index)):
        option = positions.iloc[i,4:8].values.astype('float32')
        option = np.expand_dims(option, axis=0)

        model_predictions = model_predictions.append(pd.Series(MODEL.predict_classes(option)[0]), ignore_index=True)

    # Add column to DataFrame for model predictions
    positions['ModelOutput'] = model_predictions

    # Create empty Series to collect random predictions
    random_predictions = pd.Series()  

    # Get NUM_RANDOM batches of random predictions to average
    for i in range(NUM_RANDOM):
        # Add random predictions to Series
        for j in range(len(positions.index)):
            random_predictions = random_predictions.append(pd.Series(random.randint(0, 1)), ignore_index=True)

        # Add column to DataFrame for random predictions
        positions['RandomOutput' + str(i)] = random_predictions

    return positions


def model_results(api, days, positions, symbols):
    """

    Four parameter variables, api (connects to Alpaca API), days (int), positions (DataFrame), 
    and symbols (list). Tests model predictions by getting close price data between START_DATE 
    and END_DATE for each symbol in S&P 500 and testing it against positions data to see if each
    predicted buy was profitable. Records and returns profit and performance over time.

    """

    # Set money model has made to initial investment
    model_money = INITIAL_INVESTMENT
    # Create numpy array of NaN values to keep track of performance
    model_performance = np.empty((days,1))
    model_performance[:] = np.NaN
    # Set profit for first day to 0
    model_performance[0] = 0
    # Set profit for last day to 0
    model_performance[len(model_performance) - 1] = 0

    # Get number of positions that will be invested in and amount to invest in each
    num_investments = (positions['ModelOutput'] == 1).sum()
    invest_amount = INITIAL_INVESTMENT / num_investments

    # Loop through every position
    for i in range(len(positions.index)):

        # Invest if model prediction equals 1 and expiration date is within date range
        if positions.iloc[i]['ModelOutput'] == 1 and positions.iloc[i]['ExpirationDate'] <= END_DATE:
            # Sleep for 1 second to keep API from blocking IP
            time.sleep(1)
            # Get close prices for symbol through Alpaca API
            stock_history_train = api.get_aggs(symbol=positions.iloc[i]['Symbol'], timespan='day', multiplier=1, _from=START_DATE, to=positions.iloc[i]['ExpirationDate']).df
            close_data = stock_history_train["close"]

            # Get Initial price
            init_price = close_data[0]

            # Get break even prices for the stock
            be_up = init_price + (positions.iloc[i]['BEUp'] * init_price)
            be_down = init_price - (positions.iloc[i]['BEDown'] * init_price)

            # Get costs of premiums
            put_premium = positions.iloc[i]['PutPremium']
            call_premium = positions.iloc[i]['CallPremium']

            #Get number of positions to buy
            num_positions = int(invest_amount / (call_premium + put_premium))

            made_profit = False
            done = False
            # Skip price when positions were bought
            index = 1
            # If straddle makes money sell both options immediately to take profit
            while (not done):
                if close_data[index] >= be_up:
                    straddle_profit = ((close_data[index] - be_up) * NUM_CONTRACTS) * num_positions
                    made_profit = True
                    done = True
                elif close_data[index] <= be_down and not made_profit:
                    straddle_profit = ((be_down - close_data[index]) * NUM_CONTRACTS) * num_positions
                    made_profit = True
                    done = True

                if index == len(close_data) - 1:
                    done = True
                else:
                    index += 1

            # Add profit or loss accordingly
            if made_profit:
                made_profit = False
                profit = straddle_profit
            else:
                straddle_loss = -((call_premium - put_premium) * NUM_CONTRACTS) * num_positions
                profit = straddle_loss
        
            # Add profit or loss to day when options were sold in model_performance array
            if math.isnan(model_performance[index]):
                model_performance[index] = 0
            model_performance[index] += profit

    # Add cummulative profit to model_performance and save it as model_money
    for i in range(np.shape(model_performance)[0]):
        if not math.isnan(model_performance[i]):
            model_performance[i] += model_money
            model_money = model_performance[i][0]

    return model_money, model_performance


def random_results(api, days, positions, symbols):
    """

    Four parameter variables, api (connects to Alpaca API), days (int), positions (DataFrame), 
    and symbols (list). Tests random predictions by getting close price data between START_DATE 
    and END_DATE for each symbol in S&P 500 and testing it against positions data to see if each
    predicted buy was profitable. Records and returns profit and performance over time.

    """
    # Set money random predictions have made to initial investment
    random_money = INITIAL_INVESTMENT
    # Create numpy array of NaN values to keep track of performance
    random_performance = np.empty((days,1))
    random_performance[:] = np.NaN
    # Set profit for first day to 0
    random_performance[0] = 0
    # Set profit for last day to 0
    random_performance[len(random_performance) - 1] = 0

    # Loop through every position
    for i in range(len(positions.index)):
        # Get sum for random predictions' profits
        total = 0
        for j in range(NUM_RANDOM):
            # Get number of positions that will be invested in and amount to invest in each
            num_investments = (positions['RandomOutput' + str(j)] == 1).sum()
            invest_amount = INITIAL_INVESTMENT / num_investments

            # Invest if random prediction equals 1 and expiration date is within date range
            if positions.iloc[i]['RandomOutput' + str(j)] == 1 and positions.iloc[i]['ExpirationDate'] <= END_DATE:
                # Sleep for 1 second to keep API from blocking IP
                time.sleep(1)
                # Get close prices for symbol through Alpaca API
                stock_history_train = api.get_aggs(symbol=positions.iloc[i]['Symbol'], timespan='day', multiplier=1, _from=START_DATE, to=positions.iloc[i]['ExpirationDate']).df
                close_data = stock_history_train["close"]

                # Get Initial price
                init_price = close_data[0]

                # Get break even prices for the stock
                be_up = init_price + (positions.iloc[i]['BEUp'] * init_price)
                be_down = init_price - (positions.iloc[i]['BEDown'] * init_price)

                # Get costs of premiums
                put_premium = positions.iloc[i]['PutPremium']
                call_premium = positions.iloc[i]['CallPremium']

                #Get number of positions to buy
                num_positions = int(invest_amount / (call_premium + put_premium))

                made_profit = False
                done = False
                # Skip first column
                index = 1
                # If straddle makes money sell both options immediately to take profit
                while (not done):
                    if close_data[index] >= be_up:
                        straddle_profit = ((close_data[index] - be_up) * NUM_CONTRACTS) * num_positions
                        made_profit = True
                        done = True
                    elif close_data[index] <= be_down and not made_profit:
                        straddle_profit = ((be_down - close_data[index]) * NUM_CONTRACTS) * num_positions
                        made_profit = True
                        done = True

                    if index == len(close_data) - 1:
                        done = True
                    else:
                        index += 1

                # Add profit or loss accordingly
                if made_profit:
                    made_profit = False
                    profit = straddle_profit
                else:
                    straddle_loss = -((call_premium - put_premium) * NUM_CONTRACTS) * num_positions
                    profit = straddle_loss

                # Add profit to total
                total += profit

        # Add average profit to day when options were sold in random_performance array
        if total > 0:
            if math.isnan(random_performance[index]):
                random_performance[index] = 0
            random_performance[index] += total / NUM_RANDOM

    # Add cummulative profit to random_performance and save it as random_money
    for i in range(np.shape(random_performance)[0]):
        if not math.isnan(random_performance[i]):
            random_performance[i] += random_money
            random_money = random_performance[i][0]

    return random_money, random_performance


def buyall_results(api, days, positions, symbols):
    """

    Four parameter variables, api (connects to Alpaca API), days (int), positions (DataFrame), 
    and symbols (list). Tests buying all available positions by getting close price data between 
    START_DATE and END_DATE for each symbol in S&P 500 and testing it against positions 
    data to see if each buy was profitable. Records and returns profit and performance over 
    time.

    """
    # Set money buying all positions has made to initial investment
    buyall_money = INITIAL_INVESTMENT
    # Create numpy array of NaN values to keep track of performance
    buyall_performance = np.empty((days,1))
    buyall_performance[:] = np.NaN
    # Set profit for first day to 0
    buyall_performance[0] = 0
    # Set profit for last day to 0
    buyall_performance[len(buyall_performance) - 1] = 0

    # Get number of positions that will be invested in and amount to invest in each
    num_investments = len(positions.index)
    invest_amount = INITIAL_INVESTMENT / num_investments

    # Loop through every position
    for i in range(len(positions.index)):

        # Invest if expiration date is within date range
        if positions.iloc[i]['ExpirationDate'] <= END_DATE:
            # Sleep for 1 second to keep API from blocking IP
            time.sleep(1)
            # Get close prices for symbol through Alpaca API
            stock_history_train = api.get_aggs(symbol=positions.iloc[i]['Symbol'], timespan='day', multiplier=1, _from=START_DATE, to=positions.iloc[i]['ExpirationDate']).df
            close_data = stock_history_train["close"]

            # Get Initial price
            init_price = close_data[0]

            # Get break even prices for the stock
            be_up = init_price + (positions.iloc[i]['BEUp'] * init_price)
            be_down = init_price - (positions.iloc[i]['BEDown'] * init_price)

            # Get costs of premiums
            put_premium = positions.iloc[i]['PutPremium']
            call_premium = positions.iloc[i]['CallPremium']

            #Get number of positions to buy
            num_positions = int(invest_amount / (call_premium + put_premium))

            made_profit = False
            done = False
            # Skip first column
            index = 1
            # If straddle makes money sell both options immediately to take profit
            while (not done):
                if close_data[index] >= be_up:
                    straddle_profit = ((close_data[index] - be_up) * NUM_CONTRACTS) * num_positions
                    made_profit = True
                    done = True
                elif close_data[index] <= be_down and not made_profit:
                    straddle_profit = ((be_down - close_data[index]) * NUM_CONTRACTS) * num_positions
                    made_profit = True
                    done = True

                if index == len(close_data) - 1:
                    done = True
                else:
                    index += 1

            # Add profit or loss accordingly
            if made_profit:
                made_profit = False
                profit = straddle_profit
            else:
                straddle_loss = -((call_premium - put_premium) * NUM_CONTRACTS) * num_positions
                profit = straddle_loss
        
            # Add profit or loss to day when options were sold in buyall_performance array
            if math.isnan(buyall_performance[index]):
                buyall_performance[index] = 0
            buyall_performance[index] += profit

    # Add cummulative profit to buyall_performance and save it as buyall_money
    for i in range(np.shape(buyall_performance)[0]):
        if not math.isnan(buyall_performance[i]):
            buyall_performance[i] += buyall_money
            buyall_money = buyall_performance[i][0]

    return buyall_money, buyall_performance


def market_results(api, days):
    """

    Two parameter variables, api (connects to Alpaca API) and days (int). 
    Tests market by getting close price data between START_DATE and END_DATE 
    for the S&P 500 index. Records and returns profit and performance over time.

    """
    # Set money market has made to initial investment
    market_money = INITIAL_INVESTMENT
    # Create numpy array of NaN values to keep track of performance
    market_performance = np.empty((days,1))
    # Set profit for first day to 0
    market_performance[0] = market_money

    # Set market symbol to S&P index fund
    market_symbol = 'SPY'
    # Get close prices for symbol through Alpaca API
    stock_history_train = api.get_aggs(symbol=market_symbol, timespan='day', multiplier=1, _from=START_DATE, to=END_DATE).df
    close_data = stock_history_train["close"]

    # Add profit or loss market makes each day to market_performance
    for i in range(len(close_data) - 1):
        percent_change = (close_data[i + 1] - close_data[i]) / close_data[i]

        profit = market_money * percent_change

        market_money += profit
        market_performance[i] = market_money

    # Set last day in market_performance to day before's results
    market_performance[-1] = market_money
    return market_money, market_performance


def num_days(api):
    """

    One parameter variable, api (connects to Alpaca API). Returns days (length of DataFrame of 
    SPY close prices) as market days between START_DATE and END_DATE.

    """
    market_symbol = 'SPY'
    stock_history_train = api.get_aggs(symbol=market_symbol, timespan='day', multiplier=1, _from=START_DATE, to=END_DATE).df
    close_data = stock_history_train["close"]

    days = len(close_data)

    return days


def display_results(model_final, random_final, buyall_final, market_final, model_performance, random_performance, buyall_performance, market_performance):
    """
    Eight parameter variables, model_final (float), random_final (float), 
    buyall_final (float), market_final (float), model_performance (array), 
    random_performance (array), buyall_performance (array), market_performance (array).
    Displays the initial investment value. Also displays final investment values and 
    perfomance over time for each strategy.

    """
    # Display initial investment value
    print('Initial investment: ', "$", format(INITIAL_INVESTMENT, ",.2f"), sep="")
    print('Model final: ', "$", format(model_final, ",.2f"), sep="")
    # Display final investment values
    print('Random final: ', "$", format(random_final, ",.2f"), sep="")
    print('Buy all final:', "$", format(buyall_final, ",.2f"), sep="")
    print('Market final:', "$", format(market_final, ",.2f"), sep="")

    # Create masks so NaN values can be ignored when performance is plotted
    base = np.arange(model_performance.shape[0]).reshape(model_performance.shape[0],1)
    model_mask = np.isfinite(model_performance)
    random_mask = np.isfinite(random_performance)
    buyall_mask = np.isfinite(buyall_performance)

    # Plot performances
    plt.plot(base[model_mask], model_performance[model_mask], linestyle='-', marker='o', color="red", label="model")
    plt.plot(base[random_mask], random_performance[random_mask], linestyle='-', marker='o', color="green", label="random")
    plt.plot(base[buyall_mask], buyall_performance[buyall_mask], linestyle='-', marker='o', color="purple", label="buy all")
    plt.plot(market_performance, color="blue", label="market")
    plt.legend(loc="upper right")
    plt.title('market test')
    plt.ylabel('value ($)')
    plt.xlabel('time (days)')
    plt.show()


# Run program
main()