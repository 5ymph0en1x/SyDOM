import logging
import threading
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from datetime import datetime, timedelta, timezone
from time import sleep
import time
import matplotlib.pyplot as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from utils.history import Historical_Caching
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Gives a list of timestamps from the start date to the end date
#
# startDate:     The start date as a string xxxx-xx-xx
# endDate:       The end date as a string year-month-day
# period:		 'minute', 'daily', 'weekly', or 'monthly'
# weekends:      True if weekends should be included; false otherwise
# return:        A numpy array of timestamps
def DateRange(startDate, endDate, period='minute', weekends=True):
    # The start and end date
    sd = datetime.fromtimestamp(startDate)
    ed = datetime.fromtimestamp(endDate)
    # print(startDate, endDate)
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
def DatePrevDay(startDate):
    # One day
    day = timedelta(minutes=1)
    cd = startDate - day
    return cd


# Load data from the CSV file. Note: Some systems are unable
# to give timestamps for dates before 1970. This function may
# fail on such systems.
#
# path:      The path to the file
# return:    A data frame with the parsed timestamps
def ParseData(path):
    # Read the csv file into a dataframe
    df = None
    while df is None:
        try:
            sleep(0.5)
            df = pd.read_csv(path)
        except Exception as e:
            sleep(0.5)
            pass

    df['Timestamp'] = df['Date']
    # Remove any unused columns (axis = 1 specifies fields are columns)
    df = df.drop('Date', axis=1)
    # df = df.iloc[::-1]  # CHECK THIS
    return df


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
    mpl.ylabel('Crypto Price (USD)')
    # Add the label in the upper left
    mpl.legend(loc='upper left')
    mpl.show()


# A class that predicts stock prices based on historical stock data
class Predictor:
    # Constructor
    # nPrevDays:     The number of past days to include
    #               in a sample.
    # rmodel:        The regressor model to use (sklearn)
    # nPastDays:     The number of past days in each feature
    # scaler:        The scaler object used to scale the data (sklearn)
    def __init__(self, rmodel, nPastDays, scaler=StandardScaler()):
        self.npd = nPastDays
        self.R = rmodel
        self.S = scaler
        self.D = None
        self.D_orig = None
        self.DTS = None
        self.A = None
        self.y = None
        self.targCols = None


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
        self.D_orig = D.copy()
        # self.S = StandardScaler(with_mean=True, with_std=True)
        # Keep track of old timestamps for indexing
        self.DTS = np.asarray(self.D.Timestamp.values)
        # Scale the data
        self.S.fit(self.D)
        self.D[self.D.columns] = self.S.transform(self.D)
        # Get features from the data frame
        self.A = self._ExtractFeat(self.D)
        # Get the target values and their corresponding column names
        self.y, self.targCols = self._ExtractTarg(self.D)
        # Create the regressor model and fit it
        self.R.fit(self.A, self.y)
        return True

    # Predicts values for each row of the dataframe. Can be used to
    # estimate performance of the model
    #
    # df:            The dataframe for which to make prediction
    # return:        A dataframe containing the predictions
    def PredictDF(self, df):
        # Make a local copy to prevent modifying df
        D = df.copy()
        # Scale the input data like the training data
        D[D.columns] = self.S.transform(D)
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
    # period:		 'daily', 'weekly', or 'monthly' for the time period
    #				 between predictions
    # return:        A dataframe containing the predictions or
    def PredictDate(self, startDate, endDate, period='minute'):
        # Create the range of timestamps and reverse them
        ts = DateRange(startDate, endDate, period)[::-1]
        m = ts.shape[0]
        # Prediction is based on data prior to start date
        # Get timestamp of previous day
        prevts_ = datetime.fromtimestamp(ts[-1]) - timedelta(minutes=1)
        prevts = np.asarray(prevts_.timestamp())
        # Test if there is enough data to continue
        try:
            ind = np.where(self.DTS <= prevts)[0][0]
        except IndexError:
            logger.info('Safety ON')
            ind = 0
            pass
        # There is enough data to perform prediction; allocate new data frame
        P = pd.DataFrame(np.zeros((self.D.shape[0], self.D.shape[1])), index=range(self.D.shape[0]), columns=self.D.columns)
        # Add in the timestamp column so that it can be scaled properly
        P.loc[int(m):int(self.D.shape[0]), 'Timestamp'] = self.D.loc[0:(int(self.D.shape[0] - m)), 'Timestamp']
        P.loc[0:int(m - 1), 'Timestamp'] = ts
        for i in range(self.D.shape[0] - m):
            # If the current index does not exist, repeat the last valid data
            curInd = ind + i
            if (curInd >= self.D.shape[0]):
                curInd = curInd - 1
            # Copy over the past data (already scaled)
            P.iloc[int(m + i)] = self.D_orig.xs(int(curInd))
        # for i in range(len(P)):
        #     print(datetime.datetime.fromtimestamp(P.loc[i, 'Timestamp']))
        # Scale the timestamp (other fields are 0)
        self.S.fit(P)
        P[P.columns] = self.S.transform(P)
        P = P[0:int(m * 2)]
        # B is to be the data matrix of features
        B = np.zeros((1, self._GetNumFeatures()))
        # Add extra last entries for past existing data
        # Loop until end date is reached
        # print(P)
        for i in range(m - 1, -1, -1):
            # Create one sample
            self._GetSample(B[0], i, P)
            # Predict the row of the dataframe and save it
            pred = self.R.predict(B).ravel()
            for j, k in zip(self.targCols, pred):
                P.at[i, j] = k
        # Discard extra rows needed for prediction
        # Scale the dataframe back to the original range
        P[P.columns] = self.S.inverse_transform(P)

        '''for i in range(len(P)):
            print(datetime.fromtimestamp(P.loc[i, 'Timestamp']))
        print(P)'''

        '''j = 0
        for i in P.Timestamp:
            print(dt.fromtimestamp(i))
            j += 1
            if j > 10:
                break'''

        # PlotData(P)
        P = P[0:m]
        return P

    # Test the predictors performance and
    # displays results to the screen
    #
    # D:             The dataframe for which to make prediction
    def Performance(self, df=None):
        # If no dataframe is provided, use the currently learned one
        if df is None:
            D = self.D.copy()
        else:
            self.S.fit(df)
            D = self.S.transform(df)
        # Get features from the data frame
        A = self._ExtractFeat(D)
        # Get the target values and their corresponding column names
        y, _ = self._ExtractTarg(D)
        # Begin cross validation
        ss = ShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0)

        for trn, tst in ss.split(A):
            s2 = cross_val_score(self.R, A[tst], y[tst], cv=5, scoring=make_scorer(r2_score), n_jobs=-1)

        if len(s2) > 1:
            return s2.mean()
        elif len(s2) == 1:
            logger.info(str(s2))
            return s2
        else:
            return 0


class ML_Calculus:

    def __init__(self, ws_bmex, rest, instrument, history_count, per_pred, API_key, API_secret):
        self.client = rest
        self.instrument_bmex = instrument
        self.API_key_bmex = API_key
        self.API_secret_bmex = API_secret
        self.periods_pred = per_pred - 1
        self.p_verdict = 0
        self.D = None
        self.ready = False
        self.history = Historical_Caching(ws_bmex, rest, instrument, history_count)
        self.thread = threading.Thread(target=self.Engine)
        self.R = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=25, n_jobs=-1)
        self.sp_Classic = Predictor(rmodel=self.R, nPastDays=120)
        self.logger = logging.getLogger(__name__)

    def Main(self, args):
        if (len(args) != 3 and len(args) != 4):
            return
        # Test if file exists
        try:
            open(args[0])
        except Exception as e:
            logger.error('Error opening args: ' + args[0])
            logger.error(str(e))
            return

        if (len(args) == 4):
            predPrd = args[3]
            if predPrd == 'm':
                predPrd = 'minute'
            if predPrd == 'D':
                predPrd = 'daily'
            if predPrd == 'W':
                predPrd = 'weekly'
            if predPrd == 'M':
                predPrd = 'monthly'

        try:
            # Everything looks okay; proceed with program
            # Grab the data frame
            # self.D = pand.DataFrame(index=range(self.hc))
            self.D = None
            self.D = ParseData(args[0])
            # The number of previous days of data used
            # when making a prediction

            # PlotData(D)
            s2_mean = 0
            P = None
            i = 0
            res = 0

            while res < 1:
                self.sp_Classic.Learn(self.D)
                res += 1
            while s2_mean < 0.70 and i < 3:
                # Learn the dataset and then display performance statistics
                # sp.TestPerformance()
                # Perform prediction for a specified date range
                P = self.sp_Classic.PredictDate(args[1], args[2])
                if P is None:
                    logger.info(self.instrument_bmex + ': TYPE 2 Reboot')
                    return 0, 0, 0
                s2_mean = self.sp_Classic.Performance()
                # Keep track of number of predicted results for plot
                # n = P.shape[0]
                # Append the predicted results to the actual results
                # D = P.append(D)
                # Predicted results are the first n rows
                # D.to_csv(r'xbt_m1_treated.csv', index=False)
                # PlotData(D, range(n + 1))'''
                i += 1
            # print(P)
            return i, P, s2_mean
        except Exception as e:
            logger.error(str(e))
            sleep(1)
            return 0, 0, 0

    def Engine(self):
        datetime_minute_cached = None
        fails = 0
        self.history.start_supplychain()
        while self.history.get_data_loaded() is False or self.history.get_run_completed() is False:
            logger.info(self.instrument_bmex + ': Waiting for historical data... ')
            sleep(5)
            continue
        logger.info(self.instrument_bmex + ': Starting machine learning computation...')

        while True:
            try:
                if datetime_minute_cached != datetime.utcnow().minute:  # and self.history.get_run_completed() is True:
                    start_timer = timer()

                    timestamp_ = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:00')

                    timestamp = datetime.strptime(timestamp_, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    timestamp = timestamp.timetuple()
                    start_ts = time.mktime(timestamp)

                    timestamp = datetime.strptime(timestamp_, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc) + timedelta(minutes=(self.periods_pred))
                    timestamp = timestamp.timetuple()
                    end_ts = time.mktime(timestamp)

                    naming = str(self.instrument_bmex + "_pair_m1.csv")
                    p = None
                    counter = 0

                    while isinstance(p, pd.DataFrame) is False and counter < 5:
                        i, p, s2_mean = self.Main([naming, start_ts, end_ts, 'm'])
                        counter += 1

                    if counter >= 5:
                        logger.warning(self.instrument_bmex + ': Predictor: Error p format')
                        continue

                    elif s2_mean < 0.70:
                        self.p_verdict = 0
                        logger.info(self.instrument_bmex + ' Processing Time: ' + str(round(timer() - start_timer, 5)))
                        logger.info(self.instrument_bmex + ': Machine learning : UNCONCLUSIVE !')
                        self.ready = True
                        fails += 1
                        datetime_minute_cached = datetime.utcnow().minute
                    # print('shape: ', p.shape)
                    # print(p)
                    # print(datetime.datetime.fromtimestamp(p.loc[0, 'Timestamp']))
                    # print(datetime.datetime.fromtimestamp(p.loc[1, 'Timestamp']))
                    # print(datetime.fromtimestamp(p.loc[self.periods_pred, 'Timestamp']))
                    # print(p)
                    else:
                        if np.isnan(s2_mean):
                            self.p_verdict = 0
                        else:
                            temp = self.periods_pred - 1
                            temp_1 = p.loc[temp + 1, 'Close']
                            temp_2 = p.loc[temp + 1, 'Close']
                            j = 0
                            k = 0
                            while temp >= 0:
                                p_close_tx = p.loc[temp, 'Close']
                                if temp_1 > p_close_tx:
                                    j += 1
                                elif temp_2 < p_close_tx:
                                    k += 1
                                temp_1 = temp_2 = p_close_tx
                                temp -= 1
                            if j >= self.periods_pred-1:
                                self.p_verdict = -1
                            elif k >= self.periods_pred-1:
                                self.p_verdict = 1
                            else:
                                self.p_verdict = 0
                        logger.info(self.instrument_bmex + ' Processing Time: ' + str(round(timer() - start_timer, 5)))
                        if self.p_verdict == 0:
                            logger.info(self.instrument_bmex + ' -> Machine learning : NEUTRAL !')
                        elif self.p_verdict > 0:
                            logger.info(self.instrument_bmex + ' -> Machine learning : UP !')
                        else:
                            logger.info(self.instrument_bmex + ' -> Machine learning : DOWN !')
                        self.ready = True
                        datetime_minute_cached = datetime.utcnow().minute
                        fails = 0
                    logger.info(self.instrument_bmex + ' -> Machine learning / Non-gaussian metric : ' + str(round(s2_mean * 100, 2)) + "% (iter: " + str(i) + ")")

                sleep(0.1)
            except Exception as e:
                logger.error(str(e))
                sleep(1)
                pass

    def start_ml(self):
        self.thread.daemon = True
        self.thread.start()

    def get_p_verdict(self):
        if self.p_verdict > 0:
            verdict = 1
        elif self.p_verdict < 0:
            verdict = -1
        else:
            verdict = 0
        return verdict
