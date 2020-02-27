from utils.bitmex_websocket_com import BitMEXWebsocket
from utils.autotrain import AutoTrainModel
from utils.bollinger import BollingerCalculus
from utils.rsi import RsiCalculus
from utils import bitmex_http_com as bitmex
from datetime import datetime as dt, timedelta, timezone
import logging
import pandas as pd
from signal import signal, SIGINT
import statsmodels.api as sm
from time import sleep
from sys import exit
import sys
from os import path
import os

######################################
############
API_path_bmex = 'wss://www.bitmex.com/realtime'
API_key_bmex = ''
API_secret_bmex = ''
############
instrument_bmex = "ETHUSD"
pos_size = 30
max_pos = 300
nb_cores = 12  # Number of processor cores
time_to_train = 6  # Time in server hours to retrain the model
force_training = False  # Force model training at launch
model_file = "hft_model_ETHUSD.pickle"
model_thr_1 = 0.050  # Use rsi for confirmation
model_thr_2 = 0.100  # Skip rsi
rsi_thr_upper = 80
rsi_thr_downer = 20
spread = 2  # Spread to maintain during limit order management
bb_period = 20  # Bollinger Bands periods
bb_protect = True  # Activate Bollinger Bands protection
graph_rsi = False  # Draw real-time RSI values
############
######################################

os.environ['NUMEXPR_MAX_THREADS'] = str(nb_cores)

ws_bmex = BitMEXWebsocket(endpoint=API_path_bmex, symbol=instrument_bmex, api_key=API_key_bmex, api_secret=API_secret_bmex)
client = bitmex.bitmex(test=False, api_key=API_key_bmex, api_secret=API_secret_bmex)
bb = BollingerCalculus(instrument=instrument_bmex, period=bb_period, test=False, api_key=API_key_bmex, api_secret=API_secret_bmex)
rsi = RsiCalculus(ws=ws_bmex, instrument=instrument_bmex, graph=graph_rsi)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

matrix_bmex_ticker = [None] * 6


def handler():
    logger.info(" This is the end !")
    ws_bmex.exit()
    exit(0)


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
    logger.info("--- Initiating SyDOM using BUY strategy ---")
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
    server_time = 0
    while len(ws_bmex.open_stops()) != 0:
        matrix_bmex_ticker[1] = get_ask(spread)
        matrix_bmex_ticker[2] = get_bid(spread)
        time_actual = get_time()
        if server_time != get_time():
            server_time = get_time()
            if ((time_actual > time_cached + 10000) and neutralize is False) \
                    or (bb.get_verdict() != 1 and bb.get_verdict() != -1 and neutralize is True):
                logger.info('Initial Timer Failed !')
                client.Order.Order_cancelAll().result()
                if neutralize is True:
                    return True
                else:
                    return False
            if bid_cached < matrix_bmex_ticker[2]:
                if neutralize is True:
                    if ws_bmex.open_price() > get_bid(0):
                        client.Order.Order_amend(orderID=sl_ord_number, price=get_bid(1)).result()
                    else:
                        client.Order.Order_amend(orderID=sl_ord_number, price=get_bid(0)).result()
                else:
                    client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[2]).result()
                bid_cached = matrix_bmex_ticker[2]
                logger.info(("BUY LIMIT moved @" + str(matrix_bmex_ticker[2])))
                # time_cached = get_time()
            continue
    return True


def fire_sell(neutralize=False):
    global matrix_bmex_ticker
    logger.info("--- Initiating SyDOM using SELL strategy ---")
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
    server_time = 0
    while len(ws_bmex.open_stops()) != 0:
        matrix_bmex_ticker[1] = get_ask(spread)
        matrix_bmex_ticker[2] = get_bid(spread)
        time_actual = get_time()
        if server_time != get_time():
            server_time = get_time()
            if ((time_actual > time_cached + 10000) and neutralize is False) \
                    or (bb.get_verdict() != 1 and bb.get_verdict() != -1 and neutralize is True):
                logger.info('Initial Timer Failed !')
                client.Order.Order_cancelAll().result()
                if neutralize is True:
                    return True
                else:
                    return False
            if ask_cached > matrix_bmex_ticker[1]:
                if neutralize is True:
                    if ws_bmex.open_price() < get_ask(0):
                        client.Order.Order_amend(orderID=sl_ord_number, price=get_ask(1)).result()
                    else:
                        client.Order.Order_amend(orderID=sl_ord_number, price=get_ask(0)).result()
                else:
                    client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[1]).result()
                ask_cached = matrix_bmex_ticker[1]
                logger.info(("SELL LIMIT moved @" + str(matrix_bmex_ticker[1])))
                # time_cached = get_time()
            continue
    return True


def annihilator(askP, askS, bidP, bidS, size, ts):
    resampledDF = pd.DataFrame()
    resampledDF['timestamp'] = ts
    resampledDF['size'] = size
    resampledDF.index = resampledDF['timestamp']
    resampledDF.drop(columns='timestamp', inplace=True)
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
    if X.shape == (6, 14):
        model = sm.load(model_file)
        prediction = model.predict(X)
        # print('Predictions: ', prediction.iloc[0])
        return prediction.iloc[0]
    else:
        logger.warning('Annihilator: Error X format -> ' + str(X.shape))
        return 0


def main():
    global force_training
    odbk_cached = [None]
    buff = 10
    askp = [None] * (buff+1)
    asks = [None] * (buff+1)
    bidp = [None] * (buff+1)
    bids = [None] * (buff+1)
    vol = [None] * (buff+1)
    ts = [None] * (buff+1)
    hour_cached = None
    m = 0
    counter_trial = 0

    logger.info('It began in Africa')

    if path.exists(model_file):
        model_exist = True
    else:
        model_exist = False

    if bb_protect is True:
        bb.start_bb()

    rsi.start_rsi()

    while ws_bmex.ws.sock.connected:
        try:
            DT = ws_bmex.get_instrument()['timestamp']
            dt2ts = dt.utcnow().timestamp()
            dt2retrieval = dt.strptime(DT, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            matrix_bmex_ticker[0] = int(dt2ts * 1000)
            matrix_bmex_ticker[1] = get_ask(0)
            matrix_bmex_ticker[2] = get_bid(0)
            matrix_bmex_ticker[3] = get_ask_size(0)
            matrix_bmex_ticker[4] = get_bid_size(0)
            matrix_bmex_ticker[5] = matrix_bmex_ticker[3] + matrix_bmex_ticker[4]
            hour_actual = dt2retrieval.hour
            if odbk_cached != matrix_bmex_ticker[5]:
                if (hour_actual == time_to_train and hour_cached == time_to_train-1) \
                        or model_exist is False \
                        or force_training is True:
                    logger.info('Starting AutoTraining Module... Estimated computing time: 10 minutes')
                    AutoTrainModel(model_file, instrument_bmex).start()
                    model_exist = True
                    force_training = False
                hour_cached = hour_actual
                odbk_cached = matrix_bmex_ticker[5]
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
                vol[0] = ws_bmex.recent_trades()[-1]['size']
                if m <= buff:
                    m += 1
                if m > buff and rsi.get_rsi_status():
                    verdict = annihilator(askp, asks, bidp, bids, vol, ts)
                    rsi_value = rsi.get_rsi_value()
                    if bb_protect is True:
                        bb_verdict = bb.get_verdict()
                    else:
                        bb_verdict = 0
                    if ((verdict > model_thr_1 and rsi_value > rsi_thr_upper) or verdict > model_thr_2) \
                            and bb_verdict != 1 and bb_verdict != -1:
                        if abs(ws_bmex.open_positions()) < max_pos\
                                or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() < 0):
                            logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                            # sell_action = fire_buy()
                            # if sell_action is True:
                            #     logger.info('Balance: ' + str(ws_bmex.wallet_balance()))
                    if ((verdict < -model_thr_1 and rsi_value < rsi_thr_downer) or verdict < -model_thr_2) \
                            and bb_verdict != 1 and bb_verdict != -1:
                        if abs(ws_bmex.open_positions()) < max_pos\
                                or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() > 0):
                            logger.info('SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                            # buy_action = fire_sell()
                            # if buy_action is True:
                            #     logger.info('Balance: ' + str(ws_bmex.wallet_balance()))
                    if (bb_verdict == 1 or bb_verdict == -1) and ws_bmex.open_positions() != 0:
                        logger.info('Neutralization Starting')
                        if ws_bmex.open_positions() > 0:
                            neutralization = fire_sell(neutralize=True)
                        else:
                            neutralization = fire_buy(neutralize=True)
                        if neutralization is True:
                            logger.info('Neutralization Over, standing by...')
                        else:
                            counter_trial += 1
                        if counter_trial > 5:
                            if ws_bmex.open_positions() > 0:
                                launch_order(definition='market', direction='sell', size=abs(ws_bmex.open_positions()))
                            if ws_bmex.open_positions() < 0:
                                launch_order(definition='market', direction='buy', size=abs(ws_bmex.open_positions()))
                            counter_trial = 0
                            logger.info('Forced Closure, standing by...')

        except Exception as e:
            logger.error(str(e))
            if len(ws_bmex.open_stops()) != 0:
                client.Order.Order_cancelAll().result()
            sleep(1)
            pass
            # raise


if __name__ == '__main__':
    signal(SIGINT, handler)
    main()
