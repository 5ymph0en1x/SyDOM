import logging
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

##################
time_span = 3  # Time in days covered by analysis
display_graph = False
##################


class AutoTrainModel:

    def __init__(self, filename, symbol):
        self.logger = logging
        self.filename = filename
        self.symbol = symbol
        self.v = None
        self.h = None
        self.l = None

    @jit
    def vwap(self):
        tmp1 = np.zeros_like(self.v)
        tmp2 = np.zeros_like(self.v)
        for i in range(0, len(self.v)):
            tmp1[i] = tmp1[i - 1] + self.v[i] * (self.h[i] + self.l[i]) / 2
            tmp2[i] = tmp2[i - 1] + self.v[i]
        return tmp1 / tmp2

    def run_training(self):

        url_base = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/"
        date_to_cover = []
        quote_compiled = pd.DataFrame()
        trade_compiled = pd.DataFrame()
        data1_sorted = pd.DataFrame()
        data2_sorted = pd.DataFrame()

        dt2ts = dt.utcnow()
        hour_actual = dt2ts.hour
        
        if 0 <= hour_actual < 6:
            delta = 2
        else:
            delta = 1
        
        start = dt.utcnow() - timedelta(days=delta)
        current = start
        end = start - timedelta(days=time_span-1)

        while current >= end:
            date_single = current.strftime("%Y%m%d")
            date_to_cover.append(date_single)
            current = current - timedelta(days=1)

        for i in range(len(date_to_cover)):
            date_to_download = date_to_cover[i]
            url_quote = url_base + 'quote/' + date_to_download + ".csv.gz"
            quote_downloaded = pd.read_csv(url_quote, compression='gzip')
            quote_compiled = quote_compiled.append(quote_downloaded)
            url_trade = url_base + 'trade/' + date_to_download + ".csv.gz"
            trade_downloaded = pd.read_csv(url_trade, compression='gzip')
            trade_compiled = trade_compiled.append(trade_downloaded)

        self.logger.info('quote: ' + str(len(quote_compiled)) + ' - trade: ' + str(len(trade_compiled)))

        quote_compiled.sort_values(by=['timestamp', 'symbol'], inplace=True)
        trade_compiled.sort_values(by=['timestamp', 'symbol'], inplace=True)

        quote_raw = quote_compiled.copy()
        quote_raw.sort_values(by=['symbol'], inplace=True)
        quote_raw.reset_index(drop=True, inplace=True)
        trade_raw = trade_compiled.copy()
        trade_raw.sort_values(by=['symbol'], inplace=True)
        trade_raw.reset_index(drop=True, inplace=True)
        # data_raw.set_index("timestamp", drop=False, inplace=True)
        # data_filtered = data_raw['symbol']

        for i in range(len(quote_raw)):
            sym = quote_raw['symbol'][i]
            # print(sym)
            if sym == self.symbol:
                self.logger.info(sym + ' formating...')
                data1_filtered = quote_raw.loc[quote_raw['symbol'] == sym]
                data1_sorted = data1_filtered.sort_values(by=['timestamp'], axis=0)
                # data1_sorted.to_csv('data/formated/hft/' + sym + '_quote.csv', index=False)
                data2_filtered = trade_raw.loc[trade_raw['symbol'] == sym]
                data2_sorted = data2_filtered.sort_values(by=['timestamp'], axis=0)
                # data2_sorted.to_csv('data/formated/hft/' + sym + '_trade.csv', index=False)
                self.logger.info(sym + ' done')
                break

        quote_loaded = data1_sorted.copy()
        trade_loaded = data2_sorted.copy()
        quote_loaded.sort_values(['timestamp'], inplace=True)
        trade_loaded.sort_values(['timestamp'], inplace=True)

        quote_loaded.index = pd.to_datetime(quote_loaded['timestamp'], format="%Y-%m-%dD%H:%M:%S.%f")
        quote_loaded = quote_loaded.drop(columns='timestamp')

        trade_loaded.index = pd.to_datetime(trade_loaded['timestamp'], format="%Y-%m-%dD%H:%M:%S.%f")
        trade_loaded = trade_loaded.drop(columns='timestamp')

        quote_resampled = quote_loaded.resample('500L', closed='left', label='right').last()
        quote_resampled = quote_resampled.ffill()

        trade_resampled = trade_loaded.resample('500L', closed='left', label='right').agg({"size": 'sum'})

        quote_ohlc = quote_loaded['bidPrice'].resample('500L', closed='left', label='right').ohlc().ffill()
        self.v = trade_resampled['size'].values
        self.h = quote_ohlc['high'].values
        self.l = quote_ohlc['low'].values

        vwap_ = self.vwap()
        vwap_ = pd.DataFrame({"vwap": vwap_})
        vwap_.index = quote_ohlc.index[:len(vwap_)]
        # print(vwap_.head(5))

        data_combined = quote_resampled.merge(trade_resampled, left_index=True, right_index=True)
        data_combined = data_combined.merge(quote_ohlc, left_index=True, right_index=True)
        data_combined = data_combined.merge(vwap_, left_index=True, right_index=True)
        data_combined.reset_index(drop=True, inplace=True)

        # print(data_combined.head(25))
        # break
        # print(data_combined.head(25))

        resampledDF = data_combined.copy()
        DF = pd.DataFrame()

        if display_graph is True:
            bid_trace = resampledDF[['bidPrice']]
            bid_trace.plot()
            plt.show()
            resampledDF[['vwap']].plot()
            plt.title('Plot of VWAP')
            plt.show()

        DF['deltaVtB'] = 0*(resampledDF['bidPrice']<resampledDF['bidPrice'].shift(1)) + (resampledDF['bidSize']-resampledDF['bidSize'].shift(1))*(resampledDF['bidPrice']==resampledDF['bidPrice'].shift(1)) + \
                            resampledDF['bidSize']*(resampledDF['bidPrice']>resampledDF['bidPrice'].shift(1))
        DF['deltaVtA'] = resampledDF['askSize']*(resampledDF['askPrice']<resampledDF['askPrice'].shift(1)) + (resampledDF['askSize']-resampledDF['askSize'].shift(1))*(resampledDF['askPrice']==resampledDF['askPrice'].shift(1)) + \
                            0*(resampledDF['askPrice']>resampledDF['askPrice'].shift(1))
        DF['VOI'] = DF.deltaVtB - DF.deltaVtA
        DF['DeltaVOI'] = DF.VOI.diff()

        DF['Mt'] = (resampledDF['bidPrice'] + resampledDF['askPrice']) / 2
        DF['DeltaMt'] = DF['Mt'].diff()
        resampledDF['TTV'] = DF.Mt * resampledDF['size']

        DF.dropna(inplace=True)
        X = pd.DataFrame(DF.VOI, index=DF.index)
        X = sm.add_constant(X)  ## add an intercept (beta_0) to our model
        y = DF["DeltaMt"]

        model = sm.OLS(y, X).fit()  ## sm.OLS(output, input)
        predictions = model.predict(X)

        if display_graph is True:
            sns.lmplot(x='VOI', y='DeltaMt', data=DF, fit_reg=True)
            plt.show()

        DF['OIR'] = (resampledDF['bidSize'] - resampledDF['askSize'])/(resampledDF['bidSize'] + resampledDF['askSize'])
        DF['DeltaOIR'] = DF['OIR'].diff()

        DF['TurnOver'] = resampledDF['vwap']*resampledDF['size']

        DF['TPt'] = 0
        DF.loc[DF.index[0], 'TPt'] = DF['Mt'].iloc[0]

        for i in range(1, len(DF)):
            # progress(i, len(DF)-1, ' Computing TPt')
            if resampledDF.loc[resampledDF.index[i], 'size'] == resampledDF.loc[resampledDF.index[i-1], 'size']:
                DF.loc[DF.index[i], 'TPt'] = DF.loc[DF.index[i-1], 'TPt']
            else:
                DF.loc[DF.index[i],'TPt'] = (resampledDF.loc[resampledDF.index[i],'TTV']-resampledDF.loc[resampledDF.index[i-1],'TTV'])/\
                                                        (resampledDF.loc[resampledDF.index[i],'size']-resampledDF.loc[resampledDF.index[i-1],'size'])

        # print('\n')  # new line to free the visual counter

        #Now calculate the Rt metric
        DF['Rt'] = 0
        DF['Rt'] = DF.TPt - (DF.Mt + DF.Mt.shift())/2

        if display_graph is True:
            DF.Rt.plot()
            plt.title('Plot of Rt')
            plt.show()


        ratio = []
        for i in range(1, 101):
            num = np.var(DF['Rt'].iloc[::i].diff().dropna().tolist())
            den = i*np.var(DF['Rt'].iloc[::1].diff().dropna().tolist())
            ratio.append(num/den)

        if display_graph is True:
            plt.plot(ratio)
            plt.title('Variance Ratio Test for different values of k')
            plt.ylabel('Ratio of Variances')
            plt.xlabel('Value of k')
            plt.show()

        DF['Spread'] = resampledDF['askPrice'] - resampledDF['bidPrice']

        DF['VOI0'] = DF['VOI']/DF['Spread']
        DF['OIR0'] = DF['OIR']/DF['Spread']
        DF['R0']   = DF['Rt']/DF['Spread']
        VOIFeatureList = ['VOI0']
        OIRFeatureList = ['OIR0']
        for i in range(1, 6):
            VOIString = 'VOI' + str(i)
            OIRString = 'OIR' + str(i)
            VOIFeatureList.append(VOIString)
            OIRFeatureList.append(OIRString)
            DF[VOIString] = DF['VOI'].shift(i)/DF['Spread']
            DF[OIRString] = DF['OIR'].shift(i)/DF['Spread']
        # print(DF.head(10))

        featureList = VOIFeatureList
        featureList.extend(OIRFeatureList)
        featureList.append('R0')
        DF.dropna(inplace=True)

        st = int(len(DF)/2)
        # en = int(len(DF) - 1)
        # self.logger.info("start: " + str(st) + " / end: " + str(en))
        XoutSample = DF[featureList].iloc[st:]
        XoutSample = sm.add_constant(XoutSample)  # add an intercept (beta_0) to our model

        # Build Average Price Change metric that corresponds to return over the next 20 periods
        # 20 periods
        DF['AveragePriceChange20'] = DF['Mt'].rolling(20).mean().shift(-20) - DF['Mt']
        #Run the regression now
        #Regression Analysis
        DF.dropna(inplace=True)
        X = DF[featureList]
        X = sm.add_constant(X)  # add an intercept (beta_0) to our model
        y = DF["AveragePriceChange20"]

        # Calibrate
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        model.save(self.filename)
        # predictions = model.predict(X)

        # Print out the statistics
        # print(model.summary())

        #Get the predictions
        predictions = model.predict(XoutSample)
        predictions.plot()
        # plt.show()
        plt.savefig('predict_out.png')

        self.logger.info("Model %s generated. Ending AutoTraining Module..." % self.filename)

    def start(self):
        self.run_training()
