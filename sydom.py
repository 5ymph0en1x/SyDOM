from utils.bitmex_websocket_com import BitMEXWebsocket
from utils.autotrain import AutoTrainModel
from utils import bitmex_http_com as bitmex
from datetime import datetime as dt, timedelta, timezone
import pandas as pd
import numpy as np
import numba as nb
from numba import generated_jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
from time import sleep
import sys
from os import path

######################################
############
API_path_bmex = 'wss://www.bitmex.com/realtime'
API_key_bmex = ''
API_secret_bmex = ''
############
instrument_bmex = "ETHUSD"
pos_size = 10
max_pos = 70
time_to_train = 6  # Time in server hours to retrain the model
force_training = False  # Force model training at launch
model_file = "hft_model_ETHUSD.pickle"
model_thr = 0.050
rsi_thr_upper = 70
rsi_thr_downer = 30
spread = 2  # Spread to maintain during limit order management
bb_period = 20  # Bollinger Bands period
bb_protect = True  # Activate Bollinger Bands protection
graph_rsi = False  # Draw real-time RSI values
############
######################################

ws_bmex = BitMEXWebsocket(endpoint=API_path_bmex, symbol=instrument_bmex, api_key=API_key_bmex, api_secret=API_secret_bmex)
client = bitmex.bitmex(test=False, api_key=API_key_bmex, api_secret=API_secret_bmex)
bb = bitmex.BollingerCalculus(instrument=instrument_bmex, period=bb_period, test=False, api_key=API_key_bmex, api_secret=API_secret_bmex)

ts_cached = [None]

ask_cached = 0
bid_cached = 0
chk_cached = 0
time_cached_up = 0
time_cached_down = 0
looking_for = 0
time_target = 0
prt = 0

matrix_bmex_ticker = [None] * 5
matrix_bmex_trade = [None] * 5

total_pos = 0
plt.style.use('ggplot')


# @generated_jit(fastmath=True, nopython=True)
def calc_rsi(array, deltas, avg_gain, avg_loss, p):

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


def get_rsi(array, p):

    deltas = np.append([0], np.diff(array))

    avg_gain = np.sum(deltas[1:p+1].clip(min=0)) / p
    avg_loss = -np.sum(deltas[1:p+1].clip(max=0)) / p

    array = np.empty(deltas.shape[0])
    array.fill(np.nan)

    array = calc_rsi(array, deltas, avg_gain, avg_loss, p)
    return array


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.05):
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


def retrieve_data(server_time, k):
    time_starter = server_time + timedelta(seconds=1) - timedelta(minutes=k)
    # print(time_starter)
    df = pd.DataFrame(columns=['Date', 'Close'])
    while len(df) < k:
        df = pd.DataFrame(columns=['Date', 'Close'])
        data = client.Trade.Trade_getBucketed(symbol=instrument_bmex, binSize="1m", count=k, startTime=time_starter).result()
        # print(len(data[0]))
        p = 0
        for i in data[0]:
            df.loc[p] = pd.Series({'Date': i['timestamp'], 'Close': i['close']})
            p += 1
        # print(str(len(df)) + ' / ' + str(df))
        sleep(1.0)
    # df = df[::-1]  # reverse the list
    return df


def get_ask(k):
    list_sell = []
    data = ws_bmex.market_depth()
    for batch in data:
        if batch['symbol'] == instrument_bmex:
            if batch['side'] == 'Sell':
                list_sell.append(batch)
    list_sell.sort(key=lambda i: i['id'], reverse=True)
    return list_sell[k]['price']


def get_ask_size(k):
    list_sell = []
    data = ws_bmex.market_depth()
    for batch in data:
        if batch['symbol'] == instrument_bmex:
            if batch['side'] == 'Sell':
                list_sell.append(batch)
    list_sell.sort(key=lambda i: i['id'], reverse=True)
    if len(list_sell) > k:
        return list_sell[k]['size']
    else:
        return 0


def get_all_ask_size(k):
    size_total = 0
    for i in range(0, k):
        size_total += get_ask_size(i)
    return size_total


def get_bid(k):
    list_buy = []
    data = ws_bmex.market_depth()
    for batch in data:
        if batch['symbol'] == instrument_bmex:
            if batch['side'] == 'Buy':
                list_buy.append(batch)
    list_buy.sort(key=lambda i: i['id'])
    return list_buy[k]['price']


def get_bid_size(k):
    list_buy = []
    data = ws_bmex.market_depth()
    for batch in data:
        if batch['symbol'] == instrument_bmex:
            if batch['side'] == 'Buy':
                list_buy.append(batch)
    list_buy.sort(key=lambda i: i['id'])
    if len(list_buy) > k:
        return list_buy[k]['size']
    else:
        return 0


def get_all_bid_size(k):
    size_total = 0
    for i in range(0, k):
        size_total += get_bid_size(i)
    return size_total


def get_time():
    datetime_cached = ws_bmex.recent_trades()[-1]['timestamp']
    dt2ts = dt.strptime(datetime_cached, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc).timestamp()
    time_res = int(dt2ts * 1000)  # (dt2ts - dt(1970, 1, 1)) / timedelta(seconds=1000)
    return time_res


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def launch_order(definition, direction, price=None, size=None, stoplim=None):
    resulted = 0
    if definition == 'market':
        if direction == 'sell':
            size *= -1
        resulted = client.Order.Order_new(symbol=instrument_bmex, orderQty=size, ordType='Market').result()
        return resulted[0]['orderID']
    if definition == 'limit':
        if direction == 'sell':
            size *= -1
        resulted = client.Order.Order_new(symbol=instrument_bmex, orderQty=size, ordType='Limit', price=price,
                                          execInst='ParticipateDoNotInitiate').result()
        return resulted[0]['orderID']
    if definition == 'stop_limit':
        if direction == 'sell':
            size *= -1
        resulted = client.Order.Order_new(symbol=instrument_bmex, orderQty=size, ordType='StopLimit',
                                          execInst='LastPrice',
                                          stopPx=stoplim, price=price).result()
        return resulted[0]['orderID']
    if definition == 'stop_loss':
        if direction == 'sell':
            size *= -1
        resulted = client.Order.Order_new(symbol=instrument_bmex, orderQty=size, ordType='Stop',
                                          execInst='Close, LastPrice',
                                          stopPx=price).result()
        return resulted[0]['orderID']
    if definition == 'take_profit':
        if direction == 'sell':
            size *= -1
        resulted = client.Order.Order_new(symbol=instrument_bmex, orderQty=size, ordType='Limit',
                                          execInst='Close',
                                          price=price).result()
        return resulted[0]['orderID']


def fire_buy(neutralize=False):
    global matrix_bmex_ticker
    global total_pos
    ask_snapshot = 0
    bid_snapshot = 0
    counter = 0
    print("--- Initiating SyDOM using BUY strategy ---")
    matrix_bmex_ticker[2] = get_bid(5)
    if ws_bmex.open_positions() < 0:
        if neutralize is False:
            size_official = abs(ws_bmex.open_positions()) + pos_size
        else:
            size_official = abs(ws_bmex.open_positions())
    else:
        size_official = pos_size
    sl_ord_number = launch_order(definition='limit', direction='buy', size=size_official, price=matrix_bmex_ticker[2])
    counter = 0
    while ws_bmex.open_stops() == []:
        sleep(0.1)
        counter += 1
        if counter >= 100:
            break
    bid_cached = matrix_bmex_ticker[2]
    time_cached = get_time()
    while len(ws_bmex.open_stops()) != 0:
        matrix_bmex_ticker[1] = get_ask(spread)
        matrix_bmex_ticker[2] = get_bid(spread)
        time_actual = get_time()
        if ask_snapshot != matrix_bmex_ticker[1] or bid_snapshot != matrix_bmex_ticker[2]:
            ask_snapshot = matrix_bmex_ticker[1]
            bid_snapshot = matrix_bmex_ticker[2]
            # counter += 1
            if ((counter >= 5 or time_actual > time_cached + 20000) and neutralize is False) \
                    or (bb.get_verdict() != 1 and neutralize is True):
                print('Initial Timer Failed !')
                client.Order.Order_cancelAll().result()
                if neutralize is True:
                    return True
                else:
                    return False
            if bid_cached < matrix_bmex_ticker[2]:
                if neutralize is True:
                    client.Order.Order_amend(orderID=sl_ord_number, price=get_bid(0)).result()
                else:
                    client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[2]).result()
                bid_cached = matrix_bmex_ticker[2]
                print("BUY LIMIT moved @", str(matrix_bmex_ticker[2]))
                time_cached = get_time()
                counter = 0
            continue
    return True


def fire_sell(neutralize=False):
    global matrix_bmex_ticker
    global total_pos
    ask_snapshot = 0
    bid_snapshot = 0
    counter = 0
    print("--- Initiating SyDOM using SELL strategy ---")
    matrix_bmex_ticker[1] = get_ask(5)
    if ws_bmex.open_positions() > 0:
        if neutralize is False:
            size_official = abs(ws_bmex.open_positions()) + pos_size
        else:
            size_official = abs(ws_bmex.open_positions())
    else:
        size_official = pos_size
    sl_ord_number = launch_order(definition='limit', direction='sell', size=size_official, price=matrix_bmex_ticker[1])
    counter = 0
    while ws_bmex.open_stops() == []:
        sleep(0.1)
        counter += 1
        if counter >= 100:
            break
    ask_cached = matrix_bmex_ticker[1]
    time_cached = get_time()
    while len(ws_bmex.open_stops()) != 0:
        matrix_bmex_ticker[1] = get_ask(spread)
        matrix_bmex_ticker[2] = get_bid(spread)
        time_actual = get_time()
        if ask_snapshot != matrix_bmex_ticker[1] or bid_snapshot != matrix_bmex_ticker[2]:
            ask_snapshot = matrix_bmex_ticker[1]
            bid_snapshot = matrix_bmex_ticker[2]
            # counter += 1
            if ((counter >= 5 or time_actual > time_cached + 20000) and neutralize is False) \
                    or (bb.get_verdict() != -1 and neutralize is True):
                print('Initial Timer Failed !')
                client.Order.Order_cancelAll().result()
                if neutralize is True:
                    return True
                else:
                    return False
            if ask_cached > matrix_bmex_ticker[1]:
                if neutralize is True:
                    client.Order.Order_amend(orderID=sl_ord_number, price=get_ask(0)).result()
                else:
                    client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[1]).result()
                ask_cached = matrix_bmex_ticker[1]
                print("SELL LIMIT moved @", str(matrix_bmex_ticker[1]))
                time_cached = get_time()
                counter = 0
            continue
    return True


def annihilator(askP, askS, bidP, bidS, size, ts):
    resampledDF = pd.DataFrame()
    resampledDF['timestamp'] = ts
    resampledDF['size'] = size
    resampledDF.index = resampledDF['timestamp']
    resampledDF.drop(columns='timestamp', inplace=True)
    # print(bidP)
    resampledDF['deltaVtB'] = 0
    resampledDF['deltaVtA'] = 0
    resampledDF['Mt'] = 0
    resampledDF['OIR'] = 0
    for i in range(len(ts)-1):
        resampledDF['deltaVtB'].iloc[i] = 0 * (bidP[i] < bidP[i+1]) + (bidS[i] - bidS[i+1]) * (bidP[i] == bidP[i+1]) + bidS[1] * (bidP[i] > bidP[i+1])
        resampledDF['deltaVtA'].iloc[i] = askS[i] * (askP[i] < askP[i+1]) + (askS[i] - askS[i+1]) * (askP[i] == askP[i+1]) + 0 * (askP[i] > askP[i+1])
        resampledDF['Mt'].iloc[i] = (bidP[i] + askP[i]) / 2
        resampledDF['OIR'].iloc[i] = (bidS[i] - askS[i]) / (bidS[i] + askS[i])

    resampledDF['VOI'] = resampledDF.deltaVtB - resampledDF.deltaVtA
    resampledDF['DeltaVOI'] = resampledDF.VOI.diff()
    resampledDF['TTV'] = resampledDF.Mt * resampledDF['size']
    resampledDF['TPt'] = 0
    resampledDF['TPt'] = resampledDF.Mt.copy()
    resampledDF['Rt'] = 0
    for i in range(len(ts) - 1):
        resampledDF['Rt'].iloc[i] = resampledDF.TPt.iloc[i] - (((bidP[i] + askP[i]) / 2) + ((bidP[i+1] + askP[i+1]) / 2)) / 2
    for i in range(1, len(resampledDF)):
        # progress(i, len(resampledDF) - 1, ' Computing TPt')
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
    X = sm.add_constant(X)  ## add an intercept (beta_0) to our model
    # print(X)
    model = sm.load(model_file)
    prediction = model.predict(X)
    # print('Predictions: ', prediction.iloc[0])
    return prediction.iloc[0]

buff = 10
tick_clock = 1
rsi_period = 14
depth = 10
askp = [None] * (buff+1)
asks = [None] * (buff+1)
bidp = [None] * (buff+1)
bids = [None] * (buff+1)
vol = [None] * (buff+1)
ts = [None] * (buff+1)
server_minute_cached = None
hour_cached = None
bb_verdict = 0
dom_rsi = 50
dom_size_array = [float] * (tick_clock + 3)
dom_candle_array = pd.DataFrame(columns=['open', 'high', 'low', 'close'], index=range(rsi_period + 4))
m = 0
n = 0
o = 0
size = 100
value_min = 0
value_min_cached = 0
counter_trial = 0
x_vec = np.linspace(0, 1, size+1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line1 = []

print('It began in Africa')

if path.exists(model_file):
    model_exist = True
else:
    model_exist = False

if bb_protect is True:
    bb.start_bb()

while ws_bmex.ws.sock.connected:
    try:
        DT = ws_bmex.get_instrument()['timestamp']
        dt2ts = dt.strptime(DT, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc).timestamp()
        dt2retrieval = dt.strptime(DT, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
        matrix_bmex_ticker[0] = int(dt2ts * 1000)  # (dt2ts - dt(1970, 1, 1)) / timedelta(seconds=1000)
        matrix_bmex_ticker[1] = ws_bmex.get_instrument()['askPrice']
        matrix_bmex_ticker[2] = ws_bmex.get_instrument()['bidPrice']
        matrix_bmex_ticker[3] = get_ask_size(0)
        matrix_bmex_ticker[4] = get_bid_size(0)
        hour_actual = dt2retrieval.hour
        if ts_cached != matrix_bmex_ticker[0]:
            if (hour_actual == time_to_train and hour_cached == time_to_train-1) \
                    or model_exist is False \
                    or force_training is True:
                print('Starting AutoTraining Module...')
                AutoTrainModel(model_file, instrument_bmex).start()
                model_exist = True
                force_training = False
            hour_cached = hour_actual
            ts_cached = matrix_bmex_ticker[0]
            dom_size = get_all_bid_size(depth) - get_all_ask_size(depth)
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
                    to_rsi = []
                    offset = dom_candle_array.close + abs(value_min_cached)
                    value_min = min(offset)
                    if value_min < 0:
                        to_rsi = dom_candle_array.close + abs(value_min)
                        value_min_cached += value_min
                    else:
                        to_rsi = dom_candle_array.close
                    to_rsi_ = np.array(to_rsi)
                    to_rsi_ = np.flip(to_rsi_, 0)
                    # print('Get RSI: ' + str(to_rsi_))
                    dom_rsi = get_rsi(to_rsi_, rsi_period)
                    # print('RSI: ' + str(dom_rsi[-1]))
                    if graph_rsi is True:
                        y_vec[-1] = dom_rsi[-1]
                        line1 = live_plotter(x_vec, y_vec, line1, "DOM RSI IN REAL TIME")
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
            for i in range(buff, 0, -1):
                askp[i] = askp[i-1]
                asks[i] = asks[i-1]
                bidp[i] = bidp[i-1]
                bids[i] = bids[i-1]
                ts[i] = ts[i-1]
                vol[i] = vol[i-1]
            askp[0] = matrix_bmex_ticker[1]
            asks[0] = matrix_bmex_ticker[3]
            bidp[0] = matrix_bmex_ticker[2]
            bids[0] = matrix_bmex_ticker[4]
            ts[0] = matrix_bmex_ticker[0]
            vol[0] = ws_bmex.recent_trades()[len(ws_bmex.recent_trades()) - 1]['size']
            m += 1
            if m > buff and o > rsi_period + 3:
                verdict = annihilator(askp, asks, bidp, bids, vol, ts)
                if bb_protect is True:
                    bb_verdict = bb.get_verdict()
                else:
                    bb_verdict = 0
                if verdict > model_thr and dom_rsi[-1] > rsi_thr_upper and (bb_verdict != 1 or bb_verdict != -1):
                    if abs(ws_bmex.open_positions()) < max_pos\
                            or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() > 0):
                        print('SELL ! RSI: ' + str(dom_rsi[-1]))
                        sell_action = fire_sell()
                        if sell_action is True:
                            print('Balance: ' + str(ws_bmex.wallet_balance()))
                if verdict < -model_thr and dom_rsi[-1] < rsi_thr_downer and (bb_verdict != 1 or bb_verdict != -1):
                    if abs(ws_bmex.open_positions()) < max_pos\
                            or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() < 0):
                        print('BUY ! RSI: ' + str(dom_rsi[-1]))
                        buy_action = fire_buy()
                        if buy_action is True:
                            print('Balance: ' + str(ws_bmex.wallet_balance()))
                if (bb_verdict == 1 or bb_verdict == -1) and ws_bmex.open_positions() != 0:
                    if ws_bmex.open_positions() > 0:
                        neutralization = fire_sell(neutralize=True)
                    else:
                        neutralization = fire_buy(neutralize=True)
                    if neutralization is True:
                        print('Neutralization Over, standing by...')
                    else:
                        counter_trial += 1
                    if counter_trial > 5:
                        if ws_bmex.open_positions() > 0:
                            launch_order(definition='market', direction='sell', size=abs(ws_bmex.open_positions()))
                        if ws_bmex.open_positions() < 0:
                            launch_order(definition='market', direction='buy', size=abs(ws_bmex.open_positions()))
                        counter_trial = 0
                        print('Forced Closure, standing by...')

    except Exception as e:
        print(str(e))
        if len(ws_bmex.open_stops()) != 0:
            client.Order.Order_cancelAll().result()
        sleep(1)
        pass
        # raise

    except KeyboardInterrupt:
        ws_bmex.exit()
        print(" This is the end !")
        pass
