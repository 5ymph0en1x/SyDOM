import logging
import numpy as np
import pandas as pd
from datetime import datetime as dt, timezone
import matplotlib.pyplot as plt
import threading

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
plt.style.use('ggplot')


class RsiCalculus:

    def __init__(self, ws, instrument, graph):
        self.ws_bmex = ws
        self.instrument_bmex = instrument
        self.matrix_bmex_ticker = [None] * 5
        self.depth = 15  # how deep will it go in the orderbook
        self.ts_cached = 0
        self.graph_rsi = graph
        self.rsi = 50
        self.rsi_ready = False
        self.thread = threading.Thread(target=self.compute_rsi)
        self.logger = logging.getLogger(__name__)

    def live_plotter(self, x_vec, y1_data, line1, identifier='', pause_time=0.05):
        if line1 == []:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
            # update plot label/title
            plt.ylabel('RSI Index')
            plt.title('Title: {}'.format(identifier))
            plt.show()

        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1

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

    def get_all_ask_size(self, k):
        size_total = 0
        for i in range(0, k):
            size_total += self.get_ask_size(i)
        return size_total

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

    def get_all_bid_size(self, k):
        size_total = 0
        for i in range(0, k):
            size_total += self.get_bid_size(i)
        return size_total

    # @generated_jit(fastmath=True, nopython=True)
    def calc_rsi(self, array, deltas, avg_gain, avg_loss, p):
        # Use Wilder smoothing method
        up = lambda x: x if x > 0 else 0
        down = lambda x: -x if x < 0 else 0
        i = p + 1
        # print('Delta: ' + str(list(deltas[n+1:])))
        for d in deltas[p + 1:]:
            # print(str(d))
            avg_gain = ((avg_gain * (p - 1)) + up(d)) / p
            avg_loss = ((avg_loss * (p - 1)) + down(d)) / p
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                array[i] = 100 - (100 / (1 + rs))
            else:
                array[i] = 100
            i += 1
        return array

    def get_rsi(self, array, p):
        deltas = np.append([0], np.diff(array))

        avg_gain = np.sum(deltas[1:p + 1].clip(min=0)) / p
        avg_loss = -np.sum(deltas[1:p + 1].clip(max=0)) / p

        array = np.empty(deltas.shape[0])
        array.fill(np.nan)

        array = self.calc_rsi(array, deltas, avg_gain, avg_loss, p)
        return array

    def compute_rsi(self):
        tick_clock = 1
        rsi_period = 14
        size = 100
        dom_size_array = [float] * (tick_clock + 3)
        dom_size_cached = 0
        dom_candle_array = pd.DataFrame(columns=['open', 'high', 'low', 'close'], index=range(rsi_period + 4))
        value_min_cached = 0
        x_vec = np.linspace(0, 1, size + 1)[0:-1]
        y_vec = np.random.randn(len(x_vec))
        n = 0
        o = 0
        line1 = []
        self.logger.info('Starting RSI computation...')
        while self.ws_bmex.ws.sock.connected:
            DT = self.ws_bmex.get_instrument()['timestamp']
            dt2ts = dt.strptime(DT, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc).timestamp()
            self.matrix_bmex_ticker[0] = int(dt2ts * 1000)  # (dt2ts - dt(1970, 1, 1)) / timedelta(seconds=1000)
            dom_size = self.get_all_bid_size(self.depth) - self.get_all_ask_size(self.depth)
            if dom_size != dom_size_cached:
                dom_size_cached = dom_size
                self.matrix_bmex_ticker[1] = self.ws_bmex.get_instrument()['askPrice']
                self.matrix_bmex_ticker[2] = self.ws_bmex.get_instrument()['bidPrice']
                self.matrix_bmex_ticker[3] = self.get_ask_size(0)
                self.matrix_bmex_ticker[4] = self.get_bid_size(0)
                self.ts_cached = self.matrix_bmex_ticker[0]
                dom_size_array[n] = dom_size
                # print('DOM size Array: ' + str(dom_size_array))
                n += 1
                if n >= tick_clock:
                    # dom_candle_array.loc[0]['high'] = max(dom_size_array)
                    # dom_candle_array.loc[0]['low'] = min(dom_size_array)
                    # dom_candle_array.loc[0]['open'] = dom_size_array[0]
                    n = 0
                    # print('DOM size Array: ' + str(dom_size_array))
                    dom_candle_array.iloc[0].at['close'] = dom_size_array[tick_clock-1]
                    if o >= rsi_period + 3:
                        offset = dom_candle_array.close + abs(value_min_cached)
                        value_min = min(offset)
                        if value_min < 0:
                            to_rsi = dom_candle_array.close + abs(value_min)
                            value_min_cached += value_min
                        else:
                            to_rsi = dom_candle_array.close
                        to_rsi_ = np.array(to_rsi)
                        to_rsi_ = np.flip(to_rsi_, 0)
                        dom_rsi = self.get_rsi(to_rsi_, rsi_period)
                        self.rsi = dom_rsi[-1]
                        self.rsi_ready = True
                        if self.graph_rsi is True:
                            y_vec[-1] = self.rsi
                            line1 = self.live_plotter(x_vec, y_vec, line1, "DOM RSI IN REAL TIME")
                            y_vec = np.append(y_vec[1:], 0.0)
                        dom_candle_array = dom_candle_array[:-1]
                        dom_candle_array.loc[len(dom_candle_array)] = 0
                        dom_candle_array = dom_candle_array.shift()
                    if o <= rsi_period + 3:
                        if o < rsi_period + 3:
                            # print('Adding one period: ' + str(dom_candle_array.close.to_list()))
                            dom_candle_array = dom_candle_array[:-1]
                            dom_candle_array.loc[len(dom_candle_array)] = 0
                            dom_candle_array = dom_candle_array.shift()
                        if o <= rsi_period + 3:
                            o += 1
                    # print('Rebooting')

    def start_rsi(self):
        self.thread.daemon = True
        self.thread.start()

    def get_rsi_value(self):
        return self.rsi

    def get_status(self):
        return self.rsi_ready
