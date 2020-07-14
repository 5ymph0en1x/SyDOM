from utils import bitmex_http_com as bitmex
import pandas as pd
from datetime import datetime as dt, timedelta
from time import sleep
import threading
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class BollingerCalculus:

    def __init__(self, instrument, period, test, api_key, api_secret):
        self.instrument = instrument
        self.server_time = dt.utcnow()
        self.period = period
        self.test = test
        self.api_key = api_key
        self.api_secret = api_secret
        self.thread = threading.Thread(target=self.compute_bb)
        self.df = pd.DataFrame(columns=['Date', 'Close'])
        self.verdict = 0
        self.logger = logging.getLogger(__name__)
        self.client = bitmex.bitmex(test=False, api_key=api_key, api_secret=api_secret)

    def retrieve_data(self, server_time, k):
        time_starter = server_time + timedelta(seconds=1) - timedelta(minutes=k*5)
        # print(time_starter)
        self.df = pd.DataFrame(columns=['Date', 'Close'])
        while len(self.df) < k:
            self.df = pd.DataFrame(columns=['Date', 'Close'])
            data = self.client.Trade.Trade_getBucketed(symbol=self.instrument, binSize="5m", count=k,
                                                       partial=False, startTime=time_starter).result()
            # print(len(data[0]))
            p = 0
            for i in data[0]:
                self.df.loc[p] = pd.Series({'Date': i['timestamp'], 'Close': i['close']})
                p += 1
            # print(str(len(df)) + ' / ' + str(df))
            sleep(1.0)
        # df = df[::-1]  # reverse the list
        return

    def compute_bb(self):
        self.logger.info('Starting BB computation...')
        server_minute_cached = None
        try:
            while True:
                actual_time = dt.utcnow()
                dt2ts = dt.utcnow().timestamp()
                dt2retrieval = actual_time
                server_minute = int(dt2ts // 60 % 60)
                if server_minute_cached != server_minute:
                    server_minute_cached = server_minute
                    # self.logger.info('Updating BB')
                    self.retrieve_data(dt2retrieval, self.period)
                    self.df['MA'] = self.df['Close'].rolling(window=self.period).mean()
                    self.df['STD'] = self.df['Close'].rolling(window=self.period).std()
                    self.df['Upper_Band'] = self.df['MA'] + (self.df['STD'] * 2)
                    self.df['Lower_Band'] = self.df['MA'] - (self.df['STD'] * 2)
                    self.df['Upper_Band_'] = self.df['MA'] + (self.df['STD'])
                    self.df['Lower_Band_'] = self.df['MA'] - (self.df['STD'])
                    # print(self.df.tail(5))
                    last_data = self.df[-1:]
                    # print(last_data)
                    # exit()
                    if last_data.iloc[0].at['Close'] > last_data.iloc[0].at['Upper_Band']:
                        self.logger.info('BB: Excess to the upside !')
                        self.verdict = 1
                    elif last_data.iloc[0].at['Close'] < last_data.iloc[0].at['Lower_Band']:
                        self.logger.info('BB: Excess to the downside !')
                        self.verdict = -1
                    elif last_data.iloc[0].at['Close'] > last_data.iloc[0].at['Lower_Band']\
                            and last_data.iloc[0].at['Close'] < last_data.iloc[0].at['Lower_Band_']:
                        #self.logger.info('BB: No buying allowed')
                        self.logger.info('Close: ' + str(round(last_data.iloc[0].at['Close'], 2))
                                         + ' / Upper: ' + str(round(last_data.iloc[0].at['Upper_Band'], 2))
                                         + ' / Downer: ' + str(round(last_data.iloc[0].at['Lower_Band'], 2)))
                        self.verdict = -0.5
                    elif last_data.iloc[0].at['Close'] < last_data.iloc[0].at['Upper_Band']\
                            and last_data.iloc[0].at['Close'] > last_data.iloc[0].at['Upper_Band_']:
                        # self.logger.info('BB: No selling allowed')
                        self.logger.info('Close: ' + str(round(last_data.iloc[0].at['Close'], 2))
                                         + ' / Upper: ' + str(round(last_data.iloc[0].at['Upper_Band'], 2))
                                         + ' / Downer: ' + str(round(last_data.iloc[0].at['Lower_Band'], 2)))
                        self.verdict = 0.5
                    else:
                        self.logger.info('Close: ' + str(round(last_data.iloc[0].at['Close'], 2))
                                         + ' / Upper: ' + str(round(last_data.iloc[0].at['Upper_Band'], 2))
                                         + ' / Downer: ' + str(round(last_data.iloc[0].at['Lower_Band'], 2)))
                        self.verdict = 0
                sleep(0.1)
        except Exception as e:
            self.logger.error(str(e))
            sleep(1)
            pass

    def start_bb(self):
        self.thread.daemon = True
        self.thread.start()

    def get_verdict(self):
        return self.verdict
