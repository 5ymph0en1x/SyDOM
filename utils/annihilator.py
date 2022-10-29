import logging
from time import sleep
import pandas as pd
from datetime import datetime as dt
import threading
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class Annihilator:

    def __init__(self, ws, instrument, model_file, thr_1, thr_2):
        self.ws_bmex = ws
        self.instrument_bmex = instrument
        self.model = model_file
        self.thread = threading.Thread(target=self.compute_annihilator)
        self.logger = logging.getLogger(__name__)
        self.verdict = 0
        self.thr_1 = thr_1
        self.thr_2 = thr_2
        self.ready = False

    def get_ask(self, k):
        list_sell = []
        data = self.ws_bmex.market_depth()
        for batch in data:
            if batch['symbol'] == self.instrument_bmex:
                if batch['side'] == 'Sell':
                    list_sell.append(batch)
        list_sell.sort(key=lambda i: i['id'], reverse=True)
        return list_sell[k]['price']

    def get_ask_size(self, k):
        list_sell = []
        data = self.ws_bmex.market_depth()
        for batch in data:
            if batch['symbol'] == self.instrument_bmex:
                if batch['side'] == 'Sell':
                    list_sell.append(batch)
        list_sell.sort(key=lambda i: i['id'], reverse=True)
        if len(list_sell) > k:
            return list_sell[k]['size']
        else:
            return 0

    def get_bid(self, k):
        list_buy = []
        data = self.ws_bmex.market_depth()
        for batch in data:
            if batch['symbol'] == self.instrument_bmex:
                if batch['side'] == 'Buy':
                    list_buy.append(batch)
        list_buy.sort(key=lambda i: i['id'])
        return list_buy[k]['price']

    def get_bid_size(self, k):
        list_buy = []
        data = self.ws_bmex.market_depth()
        for batch in data:
            if batch['symbol'] == self.instrument_bmex:
                if batch['side'] == 'Buy':
                    list_buy.append(batch)
        list_buy.sort(key=lambda i: i['id'])
        if len(list_buy) > k:
            return list_buy[k]['size']
        else:
            return 0

    def annihilator(self, askP, askS, bidP, bidS, size, ts):
        resampledDF = pd.DataFrame()
        resampledDF['timestamp'] = ts
        resampledDF['size'] = size
        resampledDF.index = resampledDF['timestamp']
        resampledDF.drop(columns='timestamp', inplace=True)
        resampledDF['deltaVtB'] = 0
        resampledDF['deltaVtA'] = 0
        resampledDF['Mt'] = 0
        resampledDF['OIR'] = 0
        for i in range(len(ts) - 1):
            resampledDF['deltaVtB'].iloc[i] = 0 * (bidP[i] < bidP[i + 1]) + (bidS[i] - bidS[i + 1]) * (
                        bidP[i] == bidP[i + 1]) + bidS[1] * (bidP[i] > bidP[i + 1])
            resampledDF['deltaVtA'].iloc[i] = askS[i] * (askP[i] < askP[i + 1]) + (askS[i] - askS[i + 1]) * (
                        askP[i] == askP[i + 1]) + 0 * (askP[i] > askP[i + 1])
            resampledDF['Mt'].iloc[i] = (bidP[i] + askP[i]) / 2
            resampledDF['OIR'].iloc[i] = (bidS[i] - askS[i]) / (bidS[i] + askS[i])

        resampledDF['VOI'] = resampledDF.deltaVtB - resampledDF.deltaVtA
        resampledDF['DeltaVOI'] = resampledDF.VOI.diff()
        resampledDF['TTV'] = resampledDF.Mt * resampledDF['size']
        resampledDF['TPt'] = 0
        resampledDF['TPt'] = resampledDF.Mt.copy()
        resampledDF['Rt'] = 0
        for i in range(len(ts) - 1):
            resampledDF['Rt'].iloc[i] = resampledDF.TPt.iloc[i] - (
                        ((bidP[i] + askP[i]) / 2) + ((bidP[i + 1] + askP[i + 1]) / 2)) / 2
        for i in range(1, len(resampledDF)):
            if resampledDF.loc[resampledDF.index[i], 'size'] == resampledDF.loc[resampledDF.index[i - 1], 'size']:
                resampledDF.loc[resampledDF.index[i], 'TPt'] = resampledDF.loc[resampledDF.index[i - 1], 'TPt']
            else:
                resampledDF.loc[resampledDF.index[i], 'TPt'] = (resampledDF.loc[resampledDF.index[i], 'TTV'] -
                                                                resampledDF.loc[resampledDF.index[i - 1], 'TTV']) / \
                                                               (resampledDF.loc[resampledDF.index[i], 'size'] -
                                                                resampledDF.loc[resampledDF.index[i - 1], 'size'])
        resampledDF['Spread'] = askP[0] - bidP[0]
        resampledDF['VOI0'] = resampledDF['VOI'] / resampledDF['Spread']
        resampledDF['OIR0'] = resampledDF['OIR'] / resampledDF['Spread']
        resampledDF['R0'] = resampledDF['Rt'] / resampledDF['Spread']
        VOIFeatureList = ['VOI0']
        OIRFeatureList = ['OIR0']
        for i in range(1, 6):
            VOIString = 'VOI' + str(i)
            OIRString = 'OIR' + str(i)
            VOIFeatureList.append(VOIString)
            OIRFeatureList.append(OIRString)
            resampledDF[VOIString] = resampledDF['VOI'].shift(i) / resampledDF['Spread']
            resampledDF[OIRString] = resampledDF['OIR'].shift(i) / resampledDF['Spread']
        featureList = VOIFeatureList
        featureList.extend(OIRFeatureList)
        featureList.append('R0')
        resampledDF.dropna(inplace=True)
        X = resampledDF[featureList]
        X = sm.add_constant(X, has_constant='add')  ## add an intercept (beta_0) to our model
        if X.shape == (5, 14):
            model = sm.load(self.model)
            prediction = model.predict(X)
            # print('Predictions: ', prediction.iloc[0])
            return prediction.iloc[0]
        else:
            self.logger.warning('Annihilator: Error X format -> ' + str(X.shape))
            sleep(0.05)
            return 0

    def compute_annihilator(self):
        odbk_cached = [None]
        buff = 10
        matrix_bmex_ticker = [None] * 6
        askp = [None] * buff
        asks = [None] * buff
        bidp = [None] * buff
        bids = [None] * buff
        vol = [None] * buff
        ts = [None] * buff
        m = 0
        self.logger.info('Starting Annihilator module...')
        try:
            while True:
                dt2ts = dt.utcnow().timestamp()
                matrix_bmex_ticker[0] = int(dt2ts * 1000)
                matrix_bmex_ticker[1] = self.get_ask(0)
                matrix_bmex_ticker[2] = self.get_bid(0)
                matrix_bmex_ticker[3] = self.get_ask_size(0)
                matrix_bmex_ticker[4] = self.get_bid_size(0)
                matrix_bmex_ticker[5] = matrix_bmex_ticker[3] + matrix_bmex_ticker[4]
                if odbk_cached != matrix_bmex_ticker[5]:
                    odbk_cached = matrix_bmex_ticker[5]
                    for i in range(buff - 1, 0, -1):
                        askp[i] = askp[i - 1]
                        asks[i] = asks[i - 1]
                        bidp[i] = bidp[i - 1]
                        bids[i] = bids[i - 1]
                        ts[i] = ts[i - 1]
                        vol[i] = vol[i - 1]
                    askp[0] = matrix_bmex_ticker[1]
                    asks[0] = matrix_bmex_ticker[3]
                    bidp[0] = matrix_bmex_ticker[2]
                    bids[0] = matrix_bmex_ticker[4]
                    ts[0] = matrix_bmex_ticker[0]
                    vol[0] = self.ws_bmex.recent_trades()[-1]['size']
                    if m <= buff:
                        m += 1
                    if m > buff:
                        score = self.annihilator(askp, asks, bidp, bids, vol, ts)
                        if score >= self.thr_1:
                            self.verdict = 0.5
                        if score >= self.thr_2:
                            self.verdict = 1
                        if score <= -self.thr_1:
                            self.verdict = -0.5
                        if score <= -self.thr_2:
                            self.verdict = -1
                        if self.thr_1 > score > -self.thr_1:
                            self.verdict = 0
                        self.ready = True
                sleep(0.005)

        except Exception as e:
            self.logger.error(str(e))
            sleep(1)
            raise

    def start_annihilator(self):
        self.thread.daemon = False
        self.ready = True
        self.thread.start()

    def stop_annihilator(self):
        self.ready = False
        while self.thread.is_alive() is True:
            self.thread.join()
            sleep(1.0)

    def get_verdict(self):
        return self.verdict

    def get_status(self):
        return self.ready
