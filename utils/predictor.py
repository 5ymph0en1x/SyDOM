import logging
import threading
import os
from os import path
# Used for numpy arrays
import numpy as np
# Used to read data from CSV file
import pandas as pd
# Used to convert date string to numerical value
from datetime import datetime, timedelta
from time import sleep
from utils import bitmex_http_com as bitmex
# Used to plot data
import matplotlib.pyplot as mpl
# Used to scale data
from sklearn.preprocessing import StandardScaler
# Used to perform CV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Gives a list of timestamps from the start date to the end date
#
# startDate:     The start date as a string xxxx-xx-xx
# endDate:       The end date as a string year-month-day
# period:		 'minute', 'daily', 'weekly', or 'monthly'
# weekends:      True if weekends should be included; false otherwise
# return:        A numpy array of timestamps
def DateRange(startDate, endDate, period, weekends=True):
    # The start and end date
    sd = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')
    ed = datetime.strptime(endDate, '%Y-%m-%d %H:%M:%S')
    # Invalid start and end dates
    if (sd > ed):
        raise ValueError("The start date cannot be later than the end date.")
    # One time period is a day
    if (period == 'minute'):
        prd = timedelta(minutes=1)
    if (period == 'daily'):
        prd = timedelta(1)
    # One prediction per week
    if (period == 'weekly'):
        prd = timedelta(7)
    # one prediction every 30 days ("month")
    if (period == 'monthly'):
        prd = timedelta(30)
    # The final list of timestamp data
    dates = []
    cd = sd
    while (cd <= ed):
        # If weekdays are included or it's a weekday append the current ts
        if (weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            dates.append(cd.timestamp())
        # Onto the next period
        cd = cd + prd
    # print(np.array(dates))
    return np.array(dates)


# Given a date, returns the previous day
#
# startDate:     The start date as a datetime object
# weekends:      True if weekends should counted; false otherwise
def DatePrevDay(startDate, weekends=True):
    # One day
    day = timedelta(minutes=1)
    cd = datetime.fromtimestamp(startDate)
    while (True):
        cd = cd - day
        if (weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            return cd.timestamp()
    # Should never happen
    return None


# Load data from the CSV file. Note: Some systems are unable
# to give timestamps for dates before 1970. This function may
# fail on such systems.
#
# path:      The path to the file
# return:    A data frame with the parsed timestamps
def ParseData(path):
    # Read the csv file into a dataframe
    df = pd.read_csv(path)
    # Get the date strings from the date column
    dateStr = df['Date'].values
    D = np.zeros(dateStr.shape)
    # Convert all date strings to a numeric value
    for i, j in enumerate(dateStr):
        # Date strings are of the form year-month-day
        j.replace('+00:00', '+0000')
        D[i] = datetime.strptime(j, '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None).timestamp()
    # Add the newly parsed column to the dataframe
    df['Timestamp'] = D
    # Remove any unused columns (axis = 1 specifies fields are columns)
    return df.drop('Date', axis=1)


# Given dataframe from ParseData
# plot it to the screen
#
# df:        Dataframe returned from
# p:         The position of the predicted data points
def PlotData(df, p=None):
    if (p is None):
        p = np.array([])
    # Timestamp data
    ts = df.Timestamp.values
    # Number of x tick marks
    nTicks = 10
    # Left most x value
    s = np.min(ts)
    # Right most x value
    e = np.max(ts)
    # Total range of x values
    r = e - s
    # Add some buffer on both sides
    s -= r / 5
    e += r / 5
    # These will be the tick locations on the x axis
    tickMarks = np.arange(s, e, (e - s) / nTicks)
    # Convert timestamps to strings
    strTs = [datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S') for i in tickMarks]
    mpl.figure()
    # Plots of the high and low values for the day
    mpl.plot(ts, df.High.values, color='#727272', linewidth=1.618, label='Actual')
    # Predicted data was also provided
    if (len(p) > 0):
        mpl.plot(ts[p], df.High.values[p], color='#7294AA', linewidth=1.618, label='Predicted')
    # Set the tick marks
    mpl.xticks(tickMarks, strTs, rotation='vertical')
    # Set y-axis label
    mpl.ylabel('Stock High Value (USD)')
    # Add the label in the upper left
    mpl.legend(loc='upper left')
    mpl.show()


def retrieve_data(instrument_bmex, client):
    time_starter = [0] * 4
    j = 2250
    for i in range(3):
        time_starter[i] = datetime.utcnow() - timedelta(minutes=j)
        j -= 750
    # print(time_starter)
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    data = []
    for i in range(3):
        data_temp = client.Trade.Trade_getBucketed(symbol=instrument_bmex, binSize="1m",
                                                   count=750, startTime=time_starter[i]).result()
        data.append(data_temp[0])
    j = 0
    for k in range(3):
        for i in data[k]:
            df.loc[j] = pd.Series({'Date': i['timestamp'], 'Open': i['open'], 'High': i['high'],
                                   'Low': i['low'], 'Close': i['close'], 'Volume': i['volume'],
                                   'Adj Close': i['close']})
            j += 1
    df = df[::-1]
    df.to_csv(r'pair_m1.csv', index=False)


def fetch_data(instrument_bmex, client):
    df = pd.read_csv('pair_m1.csv')
    data = client.Trade.Trade_getBucketed(symbol=instrument_bmex, binSize="1m",
                                          count=1, startTime=datetime.utcnow()-timedelta(minutes=2)).result()
    df2 = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    for i in data[0]:
        df2.loc[0] = pd.Series({'Date': i['timestamp'], 'Open': i['open'], 'High': i['high'],
                                'Low': i['low'], 'Close': i['close'], 'Volume': i['volume'],
                                'Adj Close': i['close']})
    df3 = pd.concat([df2, df], ignore_index=True)
    df3.drop(df3.tail(1).index, inplace=True)
    df3.to_csv(r'pair_m1.csv', index=False)


# A class that predicts stock prices based on historical stock data
class Predictor:
    # The (scaled) data frame
    D = None
    # Unscaled timestamp data
    DTS = None
    # The data matrix
    A = None
    # Target value matrix
    y = None
    # Corresponding columns for target values
    targCols = None
    # Number of previous days of data to use
    npd = 1
    # The regressor model
    R = None
    # Object to scale input data
    S = None

    # Constructor
    # nPrevDays:     The number of past days to include
    #               in a sample.
    # rmodel:        The regressor model to use (sklearn)
    # nPastDays:     The number of past days in each feature
    # scaler:        The scaler object used to scale the data (sklearn)
    def __init__(self, rmodel, nPastDays=1, scaler=StandardScaler()):
        self.npd = nPastDays
        self.R = rmodel
        self.S = scaler

    # Extracts features from stock market data
    #
    # D:         A dataframe from ParseData
    # ret:       The data matrix of samples
    def _ExtractFeat(self, D):
        # One row per day of stock data
        m = D.shape[0]
        # Open, High, Low, and Close for past n days + timestamp and volume
        n = self._GetNumFeatures()
        B = np.zeros([m, n])
        # Preserve order of spreadsheet
        for i in range(m - 1, -1, -1):
            self._GetSample(B[i], i, D)
        # Return the internal numpy array
        return B

    # Extracts the target values from stock market data
    #
    # D:         A dataframe from ParseData
    # ret:       The data matrix of targets and the

    def _ExtractTarg(self, D):
        # Timestamp column is not predicted
        tmp = D.drop('Timestamp', axis=1)
        # Return the internal numpy array
        return tmp.values, tmp.columns

    # Get the number of features in the data matrix
    #
    # n:         The number of previous days to include
    #           self.npd is  used if n is None
    # ret:       The number of features in the data matrix
    def _GetNumFeatures(self, n=None):
        if (n is None):
            n = self.npd
        return n * 7 + 1

    # Get the sample for a specific row in the dataframe.
    # A sample consists of the current timestamp and the data from
    # the past n rows of the dataframe
    #
    # r:         The array to fill with data
    # i:         The index of the row for which to build a sample
    # df:        The dataframe to use
    # return;    r
    def _GetSample(self, r, i, df):
        # First value is the timestamp
        r[0] = df['Timestamp'].values[i]
        # The number of columns in df
        n = df.shape[1]
        # The last valid index
        lim = df.shape[0]
        # Each sample contains the past n days of stock data; for non-existing data
        # repeat last available sample
        # Format of row:
        # Timestamp Volume Open[i] High[i] ... Open[i-1] High[i-1]... etc
        for j in range(0, self.npd):
            # Subsequent rows contain older data in the spreadsheet
            ind = i + j + 1
            # If there is no older data, duplicate the oldest available values
            if (ind >= lim):
                ind = lim - 1
            # Add all columns from row[ind]
            for k, c in enumerate(df.columns):
                # + 1 is needed as timestamp is at index 0
                r[k + 1 + n * j] = df[c].values[ind]
        return r

    # Attempts to learn the stock market data
    # given a dataframe taken from ParseData
    #
    # D:         A dataframe from ParseData
    def Learn(self, D):
        # Keep track of the currently learned data
        self.D = D.copy()
        # Keep track of old timestamps for indexing
        self.DTS = np.copy(D.Timestamp.values)
        # Scale the data
        self.D[self.D.columns] = self.S.fit_transform(self.D)
        # Get features from the data frame
        self.A = self._ExtractFeat(self.D)
        # Get the target values and their corresponding column names
        self.y, self.targCols = self._ExtractTarg(self.D)
        # Create the regressor model and fit it
        self.R.fit(self.A, self.y)

    # Predicts values for each row of the dataframe. Can be used to
    # estimate performance of the model
    #
    # df:            The dataframe for which to make prediction
    # return:        A dataframe containing the predictions
    def PredictDF(self, df):
        # Make a local copy to prevent modifying df
        D = df.copy()
        # Scale the input data like the training data
        D[D.columns] = self.S.transform()
        # Get features
        A = self._ExtractFeat(D)
        # Construct a dataframe to contain the predictions
        # Column order was saved earlier
        P = pd.DataFrame(index=range(A.shape[0]), columns=self.targCols)
        # Perform prediction
        P[P.columns] = self.R.predict(A)
        # Add the timestamp (already scaled from above)
        P['Timestamp'] = D['Timestamp'].values
        # Scale the data back to original range
        P[P.columns] = self.S.inverse_transform(P)
        return P

    # Predict the stock price during a specified time
    #
    # startDate:     The start date as a string in yyyy-mm-dd format
    # endDate:       The end date as a string yyyy-mm-dd format
    # period:		'daily', 'weekly', or 'monthly' for the time period
    #				between predictions
    # return:        A dataframe containing the predictions or
    def PredictDate(self, startDate, endDate, period='minute'):
        # Create the range of timestamps and reverse them
        ts = DateRange(startDate, endDate, period)[::-1]
        m = ts.shape[0]
        # Prediction is based on data prior to start date
        # Get timestamp of previous day
        prevts = DatePrevDay(ts[-1])
        # Test if there is enough data to continue
        try:
            ind = np.where(self.DTS == prevts)[0][0]
        except IndexError:
            return None
        # There is enough data to perform prediction; allocate new data frame
        P = pd.DataFrame(np.zeros([m, self.D.shape[1]]), index=range(m), columns=self.D.columns)
        # Add in the timestamp column so that it can be scaled properly
        P['Timestamp'] = ts
        # Scale the timestamp (other fields are 0)
        P[P.columns] = self.S.transform(P)
        # B is to be the data matrix of features
        B = np.zeros([1, self._GetNumFeatures()])
        # Add extra last entries for past existing data
        for i in range(self.npd):
            # If the current index does not exist, repeat the last valid data
            curInd = ind + i
            if (curInd >= self.D.shape[0]):
                curInd = curInd - 1
            # Copy over the past data (already scaled)
            P.loc[m + i] = self.D.loc[curInd]
        # Loop until end date is reached
        for i in range(m - 1, -1, -1):
            # Create one sample
            self._GetSample(B[0], i, P)
            # Predict the row of the dataframe and save it
            pred = self.R.predict(B).ravel()
            # Fill in the remaining fields into the respective columns
            for j, k in zip(self.targCols, pred):
                P.at[i, j] = k
        # Discard extra rows needed for prediction
        P = P[0:m]
        # Scale the dataframe back to the original range
        P[P.columns] = self.S.inverse_transform(P)
        # print(P)
        # PlotData(P)
        return P

    # Test the predictors performance and
    # displays results to the screen
    #
    # D:             The dataframe for which to make prediction
    def TestPerformance(self, df=None):
        # If no dataframe is provided, use the currently learned one
        if (df is None):
            D = self.D
        else:
            D = self.S.transform(df.copy())
        # Get features from the data frame
        A = self._ExtractFeat(D)
        # Get the target values and their corresponding column names
        y, _ = self._ExtractTarg(D)
        # Begin cross validation
        ss = ShuffleSplit(n_splits=1)
        for trn, tst in ss.split(A):
            s1 = cross_val_score(self.R, A, y, cv=3, scoring=make_scorer(r2_score))
            s2 = cross_val_score(self.R, A[tst], y[tst], cv=3, scoring=make_scorer(r2_score))
            s3 = cross_val_score(self.R, A[trn], y[trn], cv=3, scoring=make_scorer(r2_score))
            print('C-V:\t' + str(s1) + '\nTst:\t' + str(s2) + '\nTrn:\t' + str(s3))


class ML_Calculus:

    def __init__(self, instrument, API_key, API_secret):
        self.instrument_bmex = instrument
        self.API_key_bmex = API_key
        self.API_secret_bmex = API_secret
        self.p_verdict = 0
        self.thread = threading.Thread(target=self.Engine)
        # self.logger = logging.getLogger(__name__)

    # Main program
    def Main(self, args):
        if (len(args) != 3 and len(args) != 4):
            return
        # Test if file exists
        try:
            open(args[0])
        except Exception as e:
            print('Error opening file: ' + args[0])
            print(str(e))
            return
        # Test validity of start date string
        try:
            datetime.strptime(args[1], '%Y-%m-%d %H:%M:%S').timestamp()
        except Exception as e:
            print(e)
            print('Error parsing date: ' + args[1])
            return
        # Test validity of end date string
        try:
            datetime.strptime(args[2], '%Y-%m-%d %H:%M:%S').timestamp()
        except Exception as e:
            print('Error parsing date: ' + args[2])
            return
            # Test validity of final optional argument
        if (len(args) == 4):
            predPrd = args[3]
            if (predPrd == 'm'):
                predPrd = 'minute'
            if (predPrd == 'D'):
                predPrd = 'daily'
            if (predPrd == 'W'):
                predPrd = 'weekly'
            if (predPrd == 'M'):
                predPrd = 'monthly'

        # Everything looks okay; proceed with program
        # Grab the data frame
        D = ParseData(args[0])
        # The number of previous days of data used
        # when making a prediction
        numPastDays = 20
        # PlotData(D)
        # Number of neurons in the input layer
        i = numPastDays * 7 + 1
        # Number of neurons in the output layer
        o = D.shape[1] - 1
        # Number of neurons in the hidden layers
        h = int((i + o) / 2)
        # The list of layer sizes
        # layers = [('F', h), ('AF', 'tanh'), ('F', h), ('AF', 'tanh'), ('F', o)]
        # R = ANNR([i], layers, maxIter = 1000, tol = 0.01, reg = 0.001, verbose = True)
        R = KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=15)  # , n_jobs=-1)
        sp = Predictor(R, nPastDays=numPastDays)
        # Learn the dataset and then display performance statistics
        sp.Learn(D)
        # sp.TestPerformance()
        # Perform prediction for a specified date range
        P = sp.PredictDate(args[1], args[2])
        # print(P)
        # Keep track of number of predicted results for plot
        # n = P.shape[0]
        # Append the predicted results to the actual results
        # D = P.append(D)
        # Predicted results are the first n rows
        # D.to_csv(r'xbt_m1_treated.csv', index=False)
        # PlotData(D, range(n + 1))
        return (P)

    def Engine(self):
        first = True
        datetime_minute_cached = None
        client = bitmex.bitmex(test=False, api_key=self.API_key_bmex, api_secret=self.API_secret_bmex)
        logger.info('Starting machine learning computation...')
        try:
            while True:
                if datetime_minute_cached != datetime.now().minute:
                    if first is False:
                        fetch_data(self.instrument_bmex, client)
                    if first is True:
                        if path.exists('pair_m1.csv'):
                            os.remove('pair_m1.csv')
                        retrieve_data(self.instrument_bmex, client)
                        first = False
                    # print('Launching Machine learning Module...')
                    start_ts = (datetime.utcnow()+timedelta(minutes=0)).strftime("%Y-%m-%d %H:%M:00")
                    end_ts = (datetime.utcnow()+timedelta(minutes=9)).strftime("%Y-%m-%d %H:%M:00")
                    # print('Start:', start_ts, '/ End:', end_ts)
                    p = []
                    p = self.Main(['pair_m1.csv', start_ts, end_ts, 'm'])
                    p_open = p.loc[p.shape[0]-1, 'Close']
                    p_close = p.loc[0, 'Close']
                    self.p_verdict = p_open - p_close
                    if self.p_verdict < 0:
                        logger.info('Machine learning : UP !')
                    if self.p_verdict > 0:
                        logger.info('Machine learning : DOWN !')
                    datetime_minute_cached = datetime.now().minute
                sleep(0.1)
        except Exception as e:
            logger.error(str(e))
            sleep(1)
            pass

    def start_ml(self):
        self.thread.daemon = True
        self.thread.start()

    def get_p_verdict(self):
        return self.p_verdict
