from colorama import Fore
from datetime import datetime, timedelta, timezone
import logging
import math
import pandas as pd
import threading
from time import sleep
import time
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Historical_Caching:
    def __init__(self, ws, client, sym, h_c):
        self.rest = client
        self.ws_bmex = ws
        self.history_count = h_c
        self.symbol = str(sym)
        self.first = True
        self.offsetter = False
        self.Live_on = False
        self.data_loaded = False
        self.run_completed = False
        self.thread = threading.Thread(target=self.run)
        self.thread2 = threading.Thread(target=self.ohlc_live)
        self.df1_temp = None
        self.df1 = pd.DataFrame(index=range(self.history_count), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

    def load_data(self, k, l):
        iterations = math.ceil(k / 750)
        time_starter = [0] * (iterations + 1)
        b = k - 1

        for i in range(iterations):
            time_starter[i] = (datetime.utcnow() - timedelta(minutes=b * l)).astimezone(timezone.utc)
            b -= 750

        df = pd.DataFrame(index=range(self.history_count), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

        data = pd.DataFrame(index=range(self.history_count), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

        for i in range(iterations):
            data_temp = pd.Series(self.rest.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1m', count=750, partial=True, startTime=time_starter[i]).result()[0])
            if i == 0:
                data = data_temp
                continue
            data = pd.concat([data, data_temp], ignore_index=True)

        data = data.reset_index(drop=True)
        # data.to_csv(r'pair_m1_raw.csv')

        for i in range(self.history_count):
            df.iloc[i] = {'Date': data.at[i]['timestamp'].replace(tzinfo=timezone.utc).timestamp(), 'Open': data.at[i]['open'], 'High': data.at[i]['high'], 'Low': data.at[i]['low'], 'Close': data.at[i]['close'],
                 'Volume': data.at[i]['volume'], 'Adj Close': data.at[i]['close']}
            continue

        self.data_loaded = True
        return df

    def get_ask(self, k):
        list_sell = []
        data = self.ws_bmex.market_depth()
        for batch in data:
            if batch['symbol'] == self.symbol:
                if batch['side'] == 'Sell':
                    list_sell.append(batch)
        list_sell.sort(key=lambda i: i['id'], reverse=True)
        if len(list_sell) > k:
            return list_sell[k]['price']
        elif k != 0:
            return list_sell[k - 1]['price']
        else:
            return False

    def update_data(self, df):
        data_temp = []
        j = 0
        data_temp = self.rest.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1m', count=10, partial=False, reverse=True).result()
        x = data_temp[0]
        y = df.iloc[-1].Date
        df = df[::-1]
        for i in x:
            timestamp = i['timestamp'].strftime('%Y-%m-%d %H:%M:00')
            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            timestamp = timestamp.timetuple()
            ts_ = time.mktime(timestamp)
            for k in range(10):
                # print(ts_, df.iloc[k].Date)
                if ts_ == df.iloc[k].Date:
                    df.iloc[k] = {'Date': df.iloc[k].Date, 'Open': i['open'], 'High': i['high'], 'Low': i['low'], 'Close': i['close'], 'Volume': i['volume'], 'Adj Close': i['close']}
                    break
            if ts_ <= y:
                break
            else:
                j += 1

        data_temp = pd.DataFrame(x)
        data_temp = data_temp.drop(['symbol', 'trades', 'vwap', 'lastSize', 'turnover', 'homeNotional', 'foreignNotional'], axis=1)
        data_temp = data_temp.rename({'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, axis=1)
        data_temp['Adj Close'] = data_temp['Close']

        # data_temp = data_temp.drop(data_temp.head(j).index).reset_index(drop=True)
        for i in range(1, j):
            df.iloc[self.history_count - 1 - i] = {'Date': df.loc[i, 'Date'], 'Open': data_temp.loc[i, 'Open'], 'High': data_temp.loc[i, 'High'], 'Low': data_temp.loc[i, 'Low'], 'Close': data_temp.loc[i, 'Close'], 'Volume': data_temp.loc[i, 'Volume'], 'Adj Close': data_temp.loc[i, 'Close']}

        df = df[::-1]

        while self.Live_on is False:
            sleep(0.01)
        df = df.shift(-1)
        df.iloc[self.history_count - 1] = self.df1_temp
        # print('Update 1 ', len(self.df1_temp), self.df1_temp)
        # print(df.head(10), df.tail(10))

        return df

    def ohlc_live(self):
        server_minute_cached = 61
        open_1 = open_5 = open_60 = 0
        high_1 = high_5 = high_60 = 0
        low_1 = low_5 = low_60 = 100000000

        while True:
            server_minute = datetime.utcnow().minute
            if server_minute == 0:
                server_minute = 60
            ask = self.get_ask(0)
            if ask is not False:
                if open_1 == 0:
                    open_1 = ask
                if high_1 < ask:
                    high_1 = ask
                if low_1 > ask:
                    low_1 = ask
                if open_5 == 0:
                    open_5 = ask
                if high_5 < ask:
                    high_5 = ask
                if low_5 > ask:
                    low_5 = ask
                if open_60 == 0:
                    open_60 = ask
                if high_60 < ask:
                    high_60 = ask
                if low_60 > ask:
                    low_60 = ask

            if server_minute_cached != server_minute:
                self.Live_on = False
                if ask is not False:
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:00')
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc) + timedelta(minutes=1)
                    timestamp = timestamp.timetuple()
                    timestamp = time.mktime(timestamp)
                    self.df1_temp = {'Date': timestamp, 'Open': open_1, 'High': high_1, 'Low': low_1, 'Close': ask, 'Volume': 0, 'Adj Close': ask}
                    open_1 = high_1 = 0
                    low_1 = 100000000
                self.Live_on = True
                server_minute_cached = server_minute
            sleep(0.05)

    def get_history(self):
        return self.df1

    def get_live_status(self):
        return self.Live_on

    def get_update_status(self):
        return self.offsetter

    # Write Dataframe to s3 csv
    def s3_writeDfToCsv(self, df1, symbol):
        df1_ = df1.iloc[::-1]
        naming = str(symbol) + "_pair_m1.csv"
        df1_.to_csv(naming, index=False)

    def run(self):
        server_minute_cached = 61
        retry = 0
        while True:
            try:
                if self.ws_bmex.ws.sock is None:
                    logger.info('Connection status: ' + Fore.LIGHTGREEN_EX + 'Waiting for websocket' + Fore.WHITE + '...')
                    sleep(1)
                    retry += 1
                    continue
                # if first is True or (server_minute % 5 == 0 and server_minute_cached != server_minute):
                if self.first is True:
                    self.df1 = self.load_data(self.history_count, 1)
                    self.s3_writeDfToCsv(self.df1, self.symbol)
                    logger.info(str(self.symbol) + ' -> Initial historical buffering: ' + Fore.LIGHTGREEN_EX + 'OK' + Fore.WHITE + '.')
                    # logger.info(5 * '-')
                    self.first = False
                    server_minute_cached = datetime.utcnow().minute
                    self.run_completed = True
                    continue

                if server_minute_cached != datetime.utcnow().minute:
                    self.run_completed = False
                    while self.get_live_status() is False:
                        sleep(0.01)
                    self.df1 = self.update_data(self.df1)
                    self.s3_writeDfToCsv(self.df1, self.symbol)
                    self.run_completed = True
                    server_minute_cached = datetime.utcnow().minute
                    continue

                sleep(0.1)

            except Exception as e:
                logger.exception(e)
                sleep(1)
                pass

    def start_supplychain(self):
        self.warm_engine()
        self.start_engine()

    def warm_engine(self):
        self.thread2.daemon = True
        self.thread2.start()

    def start_engine(self):
        self.thread.daemon = True
        self.thread.start()

    def get_first(self):
        return self.first

    def get_data_loaded(self):
        return self.data_loaded

    def get_run_completed(self):
        return self.run_completed
