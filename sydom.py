from utils.bitmex_websocket_com import BitMEXWebsocket
from utils.autotrain import AutoTrainModel
from utils.bollinger import BollingerCalculus
from utils.rsi import RsiCalculus
from utils.annihilator import Annihilator
from utils.predictor import ML_Calculus
from utils import bitmex_http_com as bitmex
from datetime import datetime as dt, timezone
import logging
import pandas as pd
from signal import signal, SIGINT
from time import sleep
from sys import exit
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
paper_trading = False
contrarian = True  # True for the week-ends (experimental)
nb_cores = 12  # Number of processor cores
time_to_train = 6  # Time in server hours to retrain the model
force_training = False  # Force model training at launch and then exit / useful when running SyDOM the first time
model_file = "hft_model_ETHUSD.pickle"
model_thr_1 = 0.025  # Use rsi for confirmation
model_thr_2 = 0.040  # Skip rsi
rsi_thr_upper = 80
rsi_thr_downer = 20
spread = 2  # Spread to maintain during limit order management
sec_to_destroy = 120  # Number of seconds before killing a limit order : 120 if not contrarian, 10 if so
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
annihilator = Annihilator(ws=ws_bmex, instrument=instrument_bmex, model_file=model_file, thr_1=model_thr_1, thr_2=model_thr_2)
learning = ML_Calculus(instrument=instrument_bmex, API_key=API_key_bmex, API_secret=API_secret_bmex)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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


def launch_paper_order(direction, size, price):
    df = pd.DataFrame(columns=['direction', 'size', 'price'])
    df.loc[0] = pd.Series({'direction': direction, 'size': size, 'price': price})
    df.to_csv(r'order.csv', index=False)


def position_check(price):
    try:
        open('order.csv')
    except Exception as e:
        return 0
    df = pd.read_csv(r'order.csv')
    if df['direction'].iloc[0] == 'long' and df['price'].iloc[0] > price:
        try:
            open('position.csv')
        except Exception as e:
            df2 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df2.loc[0] = pd.Series({'direction': df['direction'].iloc[0], 'size': df['size'].iloc[0], 'price': df['price'].iloc[0]})
            df2.to_csv(r'position.csv', index=False)
            os.remove('order.csv')
            return 0
        df2 = pd.read_csv(r'position.csv')
        if df2['size'].iloc[0] > 0:
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            new_size = df['size'].iloc[0] + df2['size'].iloc[0]
            new_price = (df['price'].iloc[0] * df['size'].iloc[0] + df2['price'].iloc[0] * df2['size'].iloc[0]) / new_size
            df3.loc[0] = pd.Series({'direction': 'long', 'size': new_size, 'price': new_price})
            df3.to_csv(r'position.csv', index=False)
            os.remove('order.csv')
            return 0
        if df2['size'].iloc[0] < 0:
            pl = (df2['price'].iloc[0] - df['price'].iloc[0]) * df2['size'].iloc[0]
            try:
                open('pnl.csv')
            except Exception as e:
                df4 = pd.DataFrame(columns=['pnl'])
                df4.loc[0] = pd.Series({'pnl': pl})
                df4.to_csv(r'pnl.csv', index=False)
                df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
                new_size = df2['size'].iloc[0] + df['size'].iloc[0]
                df3.loc[0] = pd.Series({'direction': 'long', 'size': new_size, 'price': price})
                df3.to_csv(r'position.csv', index=False)
                os.remove('order.csv')
                return 0
            df4 = pd.read_csv(r'pnl.csv')
            pl += df4['pnl'].iloc[0]
            df4['pnl'].iloc[0] = pl
            df4.to_csv(r'pnl.csv', index=False)
            new_size = df2['size'].iloc[0] + df['size'].iloc[0]
            if new_size == 0:
                os.remove('order.csv')
                os.remove('position.csv')
                return 0
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df3.loc[0] = pd.Series({'direction': 'long', 'size': new_size, 'price': price})
            df3.to_csv(r'position.csv', index=False)
            os.remove('order.csv')
            return 0
    if df['direction'].iloc[0] == 'short' and df['price'].iloc[0] < price:
        try:
            open('position.csv')
        except Exception as e:
            df2 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df2.loc[0] = pd.Series({'direction': df['direction'].iloc[0], 'size': df['size'].iloc[0], 'price': df['price'].iloc[0]})
            df2.to_csv(r'position.csv', index=False)
            os.remove('order.csv')
            return 0
        df2 = pd.read_csv(r'position.csv')
        if df2['size'].iloc[0] < 0:
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            new_size = df['size'].iloc[0] + df2['size'].iloc[0]
            new_price = (df['price'].iloc[0] * df['size'].iloc[0] + df2['price'].iloc[0] * df2['size'].iloc[0]) / new_size
            df3.loc[0] = pd.Series({'direction': 'short', 'size': new_size, 'price': new_price})
            df3.to_csv(r'position.csv', index=False)
            os.remove('order.csv')
            return 0
        if df2['size'].iloc[0] > 0:
            pl = (df['price'].iloc[0] - df2['price'].iloc[0]) * abs(df2['size'].iloc[0])
            try:
                open('pnl.csv')
            except Exception as e:
                df4 = pd.DataFrame(columns=['pnl'])
                df4.loc[0] = pd.Series({'pnl': pl})
                df4.to_csv(r'pnl.csv', index=False)
                df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
                new_size = df2['size'].iloc[0] + df['size'].iloc[0]
                df3.loc[0] = pd.Series({'direction': 'short', 'size': new_size, 'price': price})
                df3.to_csv(r'position.csv', index=False)
                os.remove('order.csv')
                return 0
            df4 = pd.read_csv(r'pnl.csv')
            pl += df4['pnl'].iloc[0]
            df4['pnl'].iloc[0] = pl
            df4.to_csv(r'pnl.csv', index=False)
            new_size = df2['size'].iloc[0] + df['size'].iloc[0]
            if new_size == 0:
                os.remove('order.csv')
                os.remove('position.csv')
                return 0
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df3.loc[0] = pd.Series({'direction': 'short', 'size': new_size, 'price': price})
            df3.to_csv(r'position.csv', index=False)
            os.remove('order.csv')
            return 0
    return 1


def fire_buy(neutralize=False, contrarian_=False):
    matrix_bmex_ticker = [None] * 3
    logger.info("--- Initiating SyDOM using BUY strategy ---")
    matrix_bmex_ticker[2] = get_bid(5)
    order = 0
    if paper_trading is False:
        if ws_bmex.open_positions() < 0:
            if neutralize is False:
                size_official = abs(ws_bmex.open_positions()) + pos_size
            else:
                size_official = abs(ws_bmex.open_positions())
        else:
            size_official = pos_size
    if paper_trading is True:
        try:
            open('position.csv')
            df = pd.read_csv(r'position.csv')
            current_size = df['size'].iloc[0]
            if current_size < 0:
                if neutralize is False:
                    size_official = abs(current_size) + pos_size
                else:
                    size_official = abs(current_size)
            else:
                size_official = pos_size
        except Exception as e:
            size_official = pos_size
    if paper_trading is False:
        sl_ord_number = launch_order(definition='limit', direction='buy', size=size_official, price=matrix_bmex_ticker[2])
    if paper_trading is True:
        launch_paper_order(direction='long', size=size_official, price=matrix_bmex_ticker[2])
        order = 1
    counter = 0
    if paper_trading is False:
        while ws_bmex.open_stops() == []:
            sleep(0.1)
            counter += 1
            if counter >= 100:
                break
    bid_cached = matrix_bmex_ticker[2]
    time_cached = get_time()
    server_time = 0
    while ((len(ws_bmex.open_stops()) != 0 and paper_trading is False) or (order == 1 and paper_trading is True)) and (((annihilator.get_verdict() >= 0 and contrarian_ is False) or (annihilator.get_verdict() <= 0 and contrarian_ is True)) or neutralize is True):
        matrix_bmex_ticker[1] = get_ask(spread)
        matrix_bmex_ticker[2] = get_bid(spread)
        time_actual = get_time()
        if server_time != get_time():
            server_time = get_time()
            if paper_trading is True:
                order = position_check(get_ask(0))
            if ((time_actual > time_cached + sec_to_destroy * 1000) and neutralize is False) \
                    or (bb.get_verdict() == 1 and neutralize is False) \
                    or (bb.get_verdict() != 1 and bb.get_verdict() != -1 and neutralize is True):
                logger.info('Initial Timer Failed !')
                if paper_trading is False:
                    client.Order.Order_cancelAll().result()
                if paper_trading is True:
                    order = 0
                    if path.exists('order.csv'):
                        os.remove('order.csv')
                if neutralize is True:
                    return True
                else:
                    return False
            if bid_cached < matrix_bmex_ticker[2]:
                if neutralize is True:
                    if paper_trading is False:
                        if ws_bmex.open_price() > get_bid(0):
                            client.Order.Order_amend(orderID=sl_ord_number, price=get_bid(1)).result()
                        else:
                            client.Order.Order_amend(orderID=sl_ord_number, price=get_bid(0)).result()
                    if paper_trading is True:
                        df = pd.read_csv(r'order.csv')
                        open_price = df['price'].iloc[0]
                        if open_price > get_bid(0):
                            launch_paper_order(direction='long', size=size_official, price=get_bid(1))
                        else:
                            launch_paper_order(direction='long', size=size_official, price=get_bid(0))
                else:
                    if paper_trading is False:
                        client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[2]).result()
                    if paper_trading is True:
                        launch_paper_order(direction='long', size=size_official, price=matrix_bmex_ticker[2])
                bid_cached = matrix_bmex_ticker[2]
                logger.info(("BUY LIMIT moved @" + str(matrix_bmex_ticker[2])))
                # time_cached = get_time()
            continue
    if annihilator.get_verdict() < 0 and contrarian_ is False and neutralize is False:
        if paper_trading is False:
            client.Order.Order_cancelAll().result()
        if paper_trading is True:
            order = 0
            os.remove('order.csv')
        logger.info('Initial Timer Failed: Annihilator switched -> ' + str(annihilator.get_verdict()))
        return False
    if annihilator.get_verdict() > 0 and contrarian_ is True and neutralize is False:
        if paper_trading is False:
            client.Order.Order_cancelAll().result()
        if paper_trading is True:
            order = 0
            os.remove('order.csv')
        logger.info('Initial Timer Failed: Annihilator switched -> ' + str(annihilator.get_verdict()))
        return False
    if paper_trading is False:
        client.Order.Order_cancelAll().result()  # Clear remaining stops
    try:
        open('pnl.csv')
        df = pd.read_csv(r'pnl.csv')
        current_pnl = df['pnl'].iloc[0]
        logger.info('Current PnL -> ' + str(current_pnl))
    except Exception as e:
        return True
    return True


def fire_sell(neutralize=False, contrarian_=False):
    matrix_bmex_ticker = [None] * 3
    logger.info("--- Initiating SyDOM using SELL strategy ---")
    matrix_bmex_ticker[1] = get_ask(5)
    order = 0
    if paper_trading is False:
        if ws_bmex.open_positions() > 0:
            if neutralize is False:
                size_official = abs(ws_bmex.open_positions()) + pos_size
            else:
                size_official = abs(ws_bmex.open_positions())
        else:
            size_official = pos_size
    if paper_trading is True:
        try:
            open('position.csv')
            df = pd.read_csv(r'position.csv')
            current_size = df['size'].iloc[0]
            if current_size > 0:
                if neutralize is False:
                    size_official = (abs(current_size) + pos_size) * -1
                else:
                    size_official = (abs(current_size)) * -1
            else:
                size_official = pos_size * -1
        except Exception as e:
            size_official = pos_size * -1
    if paper_trading is False:
        sl_ord_number = launch_order(definition='limit', direction='sell', size=size_official, price=matrix_bmex_ticker[1])
    if paper_trading is True:
        launch_paper_order(direction='short', size=size_official, price=matrix_bmex_ticker[1])
        order = 1
    counter = 0
    if paper_trading is False:
        while ws_bmex.open_stops() == []:
            sleep(0.1)
            counter += 1
            if counter >= 100:
                break
    ask_cached = matrix_bmex_ticker[1]
    time_cached = get_time()
    server_time = 0
    while ((len(ws_bmex.open_stops()) != 0 and paper_trading is False) or (order == 1 and paper_trading is True)) and (((annihilator.get_verdict() <= 0 and contrarian_ is False) or (annihilator.get_verdict() >= 0 and contrarian_ is True)) or neutralize is True):
        matrix_bmex_ticker[1] = get_ask(spread)
        matrix_bmex_ticker[2] = get_bid(spread)
        time_actual = get_time()
        if server_time != get_time():
            server_time = get_time()
            if paper_trading is True:
                order = position_check(get_bid(0))
            if ((time_actual > time_cached + sec_to_destroy * 1000) and neutralize is False) \
                    or (bb.get_verdict() == -1 and neutralize is False) \
                    or (bb.get_verdict() != 1 and bb.get_verdict() != -1 and neutralize is True):
                logger.info('Initial Timer Failed !')
                if paper_trading is False:
                    client.Order.Order_cancelAll().result()
                if paper_trading is True:
                    order = 0
                    if path.exists('order.csv'):
                        os.remove('order.csv')
                if neutralize is True:
                    return True
                else:
                    return False
            if ask_cached > matrix_bmex_ticker[1]:
                if neutralize is True:
                    if paper_trading is False:
                        if ws_bmex.open_price() < get_ask(0):
                            client.Order.Order_amend(orderID=sl_ord_number, price=get_ask(1)).result()
                        else:
                            client.Order.Order_amend(orderID=sl_ord_number, price=get_ask(0)).result()
                    if paper_trading is True:
                        df = pd.read_csv(r'order.csv')
                        open_price = df['price'].iloc[0]
                        if open_price < get_ask(0):
                            launch_paper_order(direction='short', size=size_official, price=get_ask(1))
                        else:
                            launch_paper_order(direction='short', size=size_official, price=get_ask(0))
                else:
                    if paper_trading is False:
                        client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[1]).result()
                    if paper_trading is True:
                        launch_paper_order(direction='short', size=size_official, price=matrix_bmex_ticker[1])
                ask_cached = matrix_bmex_ticker[1]
                logger.info(("SELL LIMIT moved @" + str(matrix_bmex_ticker[1])))
                # time_cached = get_time()
            continue
    if annihilator.get_verdict() > 0 and contrarian_ is False and neutralize is False:
        if paper_trading is False:
            client.Order.Order_cancelAll().result()
        if paper_trading is True:
            order = 0
            os.remove('order.csv')
        logger.info('Initial Timer Failed: Annihilator switched -> ' + str(annihilator.get_verdict()))
        return False
    if annihilator.get_verdict() < 0 and contrarian_ is True and neutralize is False:
        if paper_trading is False:
            client.Order.Order_cancelAll().result()
        if paper_trading is True:
            order = 0
            os.remove('order.csv')
        logger.info('Initial Timer Failed: Annihilator switched -> ' + str(annihilator.get_verdict()))
        return False
    if paper_trading is False:
        client.Order.Order_cancelAll().result()  # Clear remaining stops
    try:
        open('pnl.csv')
        df = pd.read_csv(r'pnl.csv')
        current_pnl = df['pnl'].iloc[0]
        logger.info('Current PnL -> ' + str(current_pnl))
    except Exception as e:
        return True
    return True


def main():
    global force_training
    matrix_bmex_ticker = [None] * 4
    odbk_cached = [None]
    hour_cached = None
    counter_trial = 0

    logger.info('It began in Africa')

    if force_training is False:
        if path.exists(model_file):
            model_exist = True
            annihilator.start_annihilator()
        else:
            model_exist = False

        if bb_protect is True:
            bb.start_bb()

        rsi.start_rsi()
        learning.start_ml()
    else:
        model_exist = False

    while ws_bmex.ws.sock.connected:
        try:
            DT = ws_bmex.get_instrument()['timestamp']
            dt2ts = dt.utcnow().timestamp()
            dt2retrieval = dt.strptime(DT, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            matrix_bmex_ticker[0] = int(dt2ts * 1000)
            matrix_bmex_ticker[1] = get_ask_size(0)
            matrix_bmex_ticker[2] = get_bid_size(0)
            matrix_bmex_ticker[3] = matrix_bmex_ticker[1] + matrix_bmex_ticker[2]
            hour_actual = dt2retrieval.hour
            if odbk_cached != matrix_bmex_ticker[3]:
                if (hour_actual == time_to_train and hour_cached == time_to_train-1) \
                        or model_exist is False \
                        or force_training is True:
                    logger.info('Starting AutoTraining Module... Estimated computing time: 10 minutes')
                    AutoTrainModel(model_file, instrument_bmex).start()
                    model_exist = True
                    if force_training is True:
                        logger.info(" This is the end !")
                        ws_bmex.exit()
                        exit(0)
                    if annihilator.get_status() is False:
                        annihilator.start_annihilator()
                hour_cached = hour_actual
                odbk_cached = matrix_bmex_ticker[3]
                if annihilator.get_status() and rsi.get_status():
                    verdict = annihilator.get_verdict()
                    rsi_value = rsi.get_rsi_value()
                    ml_verdict = learning.get_p_verdict()
                    if bb_protect is True:
                        bb_verdict = bb.get_verdict()
                    else:
                        bb_verdict = 0
                    if ((verdict >= 0.5 and rsi_value > rsi_thr_upper) or verdict == 1) \
                            and bb_verdict != 1 and bb_verdict != -1 and ml_verdict < 0:
                        if paper_trading is False:
                            if abs(ws_bmex.open_positions()) < max_pos\
                                    or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() < 0):
                                if contrarian is False:
                                    logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                    buy_action = fire_buy()
                                elif abs(ws_bmex.open_positions()) < max_pos:
                                    logger.info('CONTRARIAN SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                    buy_action = fire_sell(contrarian_=True)
                                if buy_action is True:
                                    logger.info('Balance: ' + str(ws_bmex.wallet_balance()))
                        if paper_trading is True:
                            try:
                                open('position.csv')
                                df = pd.read_csv(r'position.csv')
                                current_size = df['size'].iloc[0]
                                if abs(current_size) < max_pos \
                                        or (abs(current_size) >= max_pos and current_size < 0):
                                    if contrarian is False:
                                        logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                            round(verdict, 3)))
                                        fire_buy()
                                    elif abs(current_size) < max_pos:
                                        logger.info(
                                            'CONTRARIAN SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                                round(verdict, 3)))
                                        fire_sell(contrarian_=True)
                            except Exception as e:
                                if contrarian is False:
                                    logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                        round(verdict, 3)))
                                    fire_buy()
                                else:
                                    logger.info(
                                        'CONTRARIAN SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                            round(verdict, 3)))
                                    fire_sell(contrarian_=True)
                    if ((verdict <= -0.5 and rsi_value < rsi_thr_downer) or verdict == -1) \
                            and bb_verdict != 1 and bb_verdict != -1 and ml_verdict > 0:
                        if paper_trading is False:
                            if abs(ws_bmex.open_positions()) < max_pos\
                                    or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() > 0):
                                if contrarian is False:
                                    logger.info('SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                    sell_action = fire_sell()
                                elif abs(ws_bmex.open_positions()) < max_pos:
                                    logger.info('CONTRARIAN BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                    sell_action = fire_buy(contrarian_=True)
                                if sell_action is True:
                                    logger.info('Balance: ' + str(ws_bmex.wallet_balance()))
                        if paper_trading is True:
                            try:
                                open('position.csv')
                                df = pd.read_csv(r'position.csv')
                                current_size = df['size'].iloc[0]
                                if abs(current_size) < max_pos \
                                        or (abs(current_size) >= max_pos and current_size < 0):
                                    if contrarian is False:
                                        logger.info('SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                            round(verdict, 3)))
                                        fire_sell()
                                    elif abs(current_size) < max_pos:
                                        logger.info(
                                            'CONTRARIAN BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                                round(verdict, 3)))
                                        fire_buy(contrarian_=True)
                            except Exception as e:
                                if contrarian is False:
                                    logger.info('SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                        round(verdict, 3)))
                                    fire_sell()
                                else:
                                    logger.info(
                                        'CONTRARIAN BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                            round(verdict, 3)))
                                    fire_buy(contrarian_=True)
                    if paper_trading is False and (bb_verdict == 1 or bb_verdict == -1) and ws_bmex.open_positions() != 0:
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
                    if paper_trading is True and (bb_verdict == 1 or bb_verdict == -1):
                        try:
                            open('position.csv')
                            df = pd.read_csv(r'position.csv')
                            current_size = df['size'].iloc[0]
                            if current_size != 0:
                                logger.info('Neutralization Starting')
                                if current_size > 0:
                                    neutralization = fire_sell(neutralize=True)
                                else:
                                    neutralization = fire_buy(neutralize=True)
                                if neutralization is True:
                                    logger.info('Neutralization Over, standing by...')
                                    os.remove('position.csv')
                        except Exception as e:
                            pass

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
