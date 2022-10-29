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
import os
from colorama import Fore
from bravado.exception import HTTPBadRequest, HTTPServiceUnavailable, HTTPServerError
import multiprocessing

######################################
############
API_path_bmex = 'wss://ws.bitmex.com/realtime'
API_key_bmex = '***'
API_secret_bmex = '***'
############
instrument_bmex = "ETHUSD"
pos_size = 10
max_pos = 50
history_periods = 4320  # Number of periods to be analyzed by the ML module (in minutes)
predicted_periods = 3  # Number of periods to be predicted by the ML module (in minutes)
paper_trading = True  # True for simulating trading behaviour
contrarian = False  # True for the weekends (experimental)
time_to_train = 6  # Time in server hours to retrain the model
skip_initial_training = False
model_file = str(instrument_bmex) + "_hft_model.pickle"
rsi_thr_upper = 80
rsi_thr_downer = 20
spread = 2  # Spread to maintain during limit order management
sec_to_destroy = 300  # Number of seconds before killing a limit order
quotes_to_destroy = 15  # Number of upticks/downticks received before cancelling a limit order
bb_period = 20  # Bollinger Bands periods
bb_protect = False  # Activate Bollinger Bands protection
graph_rsi = False  # Draw real-time RSI values
############
######################################

print("""
            
            ███████╗██╗   ██╗██████╗  ██████╗ ███╗   ███╗
            ██╔════╝╚██╗ ██╔╝██╔══██╗██╔═══██╗████╗ ████║
            ███████╗ ╚████╔╝ ██║  ██║██║   ██║██╔████╔██║
            ╚════██║  ╚██╔╝  ██║  ██║██║   ██║██║╚██╔╝██║
            ███████║   ██║   ██████╔╝╚██████╔╝██║ ╚═╝ ██║
            ╚══════╝   ╚═╝   ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
                                                                              

""")

os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

ws_bmex = BitMEXWebsocket(endpoint=API_path_bmex, symbol=instrument_bmex, api_key=API_key_bmex, api_secret=API_secret_bmex)
client = bitmex.bitmex(test=False, api_key=API_key_bmex, api_secret=API_secret_bmex)
bb = BollingerCalculus(instrument=instrument_bmex, period=bb_period, test=False, api_key=API_key_bmex, api_secret=API_secret_bmex)
rsi = RsiCalculus(ws=ws_bmex, instrument=instrument_bmex, graph=graph_rsi)
learning = ML_Calculus(ws_bmex=ws_bmex, rest=client, instrument=instrument_bmex, history_count=history_periods, per_pred=predicted_periods, API_key=API_key_bmex, API_secret=API_secret_bmex)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(signum, frame):
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


def launch_order(instr_bmex, definition, direction, price=None, size=None, stoplim=None):
    resulted = 0
    if definition == 'market':
        if direction == 'sell':
            if size > 0:
                size *= -1
        resulted = client.Order.Order_new(symbol=instr_bmex, orderQty=size, ordType='Market').result()
        return resulted[0]['orderID']
    if definition == 'limit':
        if direction == 'sell':
            if size > 0:
                size *= -1
        resulted = client.Order.Order_new(symbol=instr_bmex, orderQty=size, ordType='Limit', price=price,
                                          execInst='ParticipateDoNotInitiate').result()
        return resulted[0]['orderID']
    if definition == 'stop_limit':
        if direction == 'sell':
            if size > 0:
                size *= -1
        resulted = client.Order.Order_new(symbol=instr_bmex, orderQty=size, ordType='StopLimit',
                                          execInst='LastPrice',
                                          stopPx=stoplim, price=price).result()
        return resulted[0]['orderID']
    if definition == 'stop_loss':
        if direction == 'sell':
            if size > 0:
                size *= -1
        resulted = client.Order.Order_new(symbol=instr_bmex, orderQty=size, ordType='Stop',
                                          execInst='Close, LastPrice',
                                          stopPx=price).result()
        return resulted[0]['orderID']
    if definition == 'take_profit':
        if direction == 'sell':
            size *= -1
        resulted = client.Order.Order_new(symbol=instr_bmex, orderQty=size, ordType='Limit',
                                          execInst='Close',
                                          price=price).result()
        return resulted[0]['orderID']

    return resulted


def launch_paper_order(instr_bmex, direction, size, price):
    df = pd.DataFrame(columns=['direction', 'size', 'price'])
    df.loc[0] = pd.Series({'direction': direction, 'size': size, 'price': price})
    naming = str(instr_bmex) + '_order.csv'
    df.to_csv(naming, index=False)


def get_profit(instr_bmex):
    naming = str(instr_bmex) + '_position.csv'
    try:
        open(naming)
    except Exception as e:
        return None
    df = pd.read_csv(naming)
    base = df['price'].iloc[0]
    if df['direction'].iloc[0] == 'long':
        profit = df['size'].iloc[0] * (1/base - 1/get_bid(0)) * get_bid(0)
        return profit
    elif df['direction'].iloc[0] == 'short':
        profit = df['size'].iloc[0] * (1/base - 1/get_ask(0)) * get_ask(0)
        return profit
    else:
        return 0


def position_check(instr_bmex, price, neutralize=False, market=False):

    naming_order = str(instr_bmex) + '_order.csv'
    naming_pos = str(instr_bmex) + '_position.csv'
    try:
        open(naming_order)
    except Exception as e:
        return 0
    df = pd.read_csv(naming_order)
    if df['direction'].iloc[0] == 'long' and (df['price'].iloc[0] > price or market is True):
        # print('Validate Long')
        try:
            open(naming_pos)
        except Exception as e:
            df2 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df2.loc[0] = pd.Series(
                {'direction': df['direction'].iloc[0], 'size': df['size'].iloc[0], 'price': df['price'].iloc[0]})
            df2.to_csv(naming_pos, index=False)
            os.remove(naming_order)
            return 0
        df2 = pd.read_csv(naming_pos)
        if df2['size'].iloc[0] > 0:
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            new_size = df['size'].iloc[0] + df2['size'].iloc[0]
            new_price = (df['price'].iloc[0] * df['size'].iloc[0] + df2['price'].iloc[0] * df2['size'].iloc[0]) / new_size
            df3.loc[0] = pd.Series({'direction': 'long', 'size': new_size, 'price': new_price})
            df3.to_csv(naming_pos, index=False)
            os.remove(naming_order)
            return 0
        if df2['size'].iloc[0] < 0:
            pl = get_profit(instr_bmex)
            try:
                open('pnl.csv')
            except Exception as e:
                df4 = pd.DataFrame(columns=['pnl'])
                df4.loc[0] = pd.Series({'pnl': pl})
                df4.to_csv(r'pnl.csv', index=False)
                os.remove(naming_order)
                if neutralize is True:
                    os.remove(naming_pos)
                    return 0
                if neutralize is False:
                    df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
                    new_size = df2['size'].iloc[0] + df['size'].iloc[0]
                    df3.loc[0] = pd.Series({'direction': 'long', 'size': new_size, 'price': price})
                    df3.to_csv(naming_pos, index=False)
                    return 0
            df4 = pd.read_csv(r'pnl.csv')
            pl += df4['pnl'].iloc[0]
            df4['pnl'].iloc[0] = pl
            df4.to_csv(r'pnl.csv', index=False)
            new_size = df2['size'].iloc[0] + df['size'].iloc[0]
            if new_size == 0:
                os.remove(naming_order)
                os.remove(naming_pos)
                return 0
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df3.loc[0] = pd.Series({'direction': 'long', 'size': new_size, 'price': price})
            df3.to_csv(naming_pos, index=False)
            os.remove(naming_order)
            return 0
    if df['direction'].iloc[0] == 'short' and (df['price'].iloc[0] < price or market is True):
        # print('Validate Short')
        try:
            open(naming_pos)
        except Exception as e:
            df2 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df2.loc[0] = pd.Series(
                {'direction': df['direction'].iloc[0], 'size': df['size'].iloc[0], 'price': df['price'].iloc[0]})
            df2.to_csv(naming_pos, index=False)
            os.remove(naming_order)
            return 0
        df2 = pd.read_csv(naming_pos)
        if df2['size'].iloc[0] < 0:
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            new_size = df['size'].iloc[0] + df2['size'].iloc[0]
            new_price = (df['price'].iloc[0] * df['size'].iloc[0] + df2['price'].iloc[0] * df2['size'].iloc[0]) / new_size
            df3.loc[0] = pd.Series({'direction': 'short', 'size': new_size, 'price': new_price})
            df3.to_csv(naming_pos, index=False)
            os.remove(naming_order)
            return 0
        if df2['size'].iloc[0] > 0:
            pl = get_profit(instr_bmex)
            try:
                open('pnl.csv')
            except Exception as e:
                df4 = pd.DataFrame(columns=['pnl'])
                df4.loc[0] = pd.Series({'pnl': pl})
                df4.to_csv(r'pnl.csv', index=False)
                os.remove(naming_order)
                if neutralize is True:
                    os.remove(naming_pos)
                    return 0
                if neutralize is False:
                    df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
                    new_size = df2['size'].iloc[0] + df['size'].iloc[0]
                    df3.loc[0] = pd.Series({'direction': 'short', 'size': new_size, 'price': price})
                    df3.to_csv(naming_pos, index=False)
                    return 0
            df4 = pd.read_csv(r'pnl.csv')
            pl += df4['pnl'].iloc[0]
            df4['pnl'].iloc[0] = pl
            df4.to_csv(r'pnl.csv', index=False)
            new_size = df2['size'].iloc[0] + df['size'].iloc[0]
            if new_size == 0:
                os.remove(naming_order)
                os.remove(naming_pos)
                return 0
            df3 = pd.DataFrame(columns=['direction', 'size', 'price'])
            df3.loc[0] = pd.Series({'direction': 'short', 'size': new_size, 'price': price})
            df3.to_csv(naming_pos, index=False)
            os.remove(naming_order)
            return 0
    return 1


def fire_buy(instr_bmex, pos_size, contrarian_=False, neutralize=False):

    naming_order = str(instr_bmex) + '_order.csv'
    naming_pos = str(instr_bmex) + '_position.csv'

    try:
        matrix_bmex_ticker = [None] * 3
        if neutralize is False:
            logger.info(instr_bmex + ": --- Initiating BUY strategy ---")
        else:
            if paper_trading is True:
                try:
                    open(naming_order)
                    os.remove(naming_order)
                except Exception as e:
                    try:
                        open(naming_pos)
                        pass
                    except Exception as e:
                        logger.info(instr_bmex + ": --- Stopping Neutralizing Mode ---")
                        return True
                logger.info(instr_bmex + ": --- Initiating Neutralizing Mode ---")
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
        else:
            try:
                open(naming_pos)
                df = pd.read_csv(naming_pos)
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
                pass
        if paper_trading is False:
            sl_ord_number = launch_order(instr_bmex, definition='limit', direction='buy', size=size_official, price=matrix_bmex_ticker[2])
        if paper_trading is True:
            launch_paper_order(instr_bmex, direction='long', size=size_official, price=matrix_bmex_ticker[2])
            order = 1
        counter = 0
        if paper_trading is False:
            try:
                while ws_bmex.open_orders_raw(instr_bmex) == [] or ws_bmex.open_orders_raw(instr_bmex)[len(ws_bmex.open_orders_raw(instr_bmex)) - 1]['ordStatus'] != "New":
                    sleep(0.1)
                    counter += 1
                    if counter >= 100:
                        break
            except Exception as e:
                logger.exception(e)
                pass
        bid_cached = matrix_bmex_ticker[2]
        time_cached = get_time()
        counter = 0
        try:
            while (order == 1 and paper_trading is True) or \
                    (ws_bmex.open_orders_raw(instr_bmex) != [] and ws_bmex.open_orders_raw(instr_bmex)[len(ws_bmex.open_orders_raw(instr_bmex)) - 1]['ordStatus'] == "New" and paper_trading is False):
                matrix_bmex_ticker[1] = get_ask(spread - 1)
                matrix_bmex_ticker[2] = get_bid(spread - 1)
                time_actual = get_time()
                if ((learning.get_p_verdict() == -1 and contrarian_ is False) or (learning.get_p_verdict() > 0 and contrarian_ is True)) and neutralize is False:
                    if paper_trading is False:
                        client.Order.Order_cancelAll().result()
                    else:
                        # order = 0
                        os.remove(naming_order)
                    logger.info(instr_bmex + ': Initial bullish micro-trend just reversed !')
                    return False
                if paper_trading is True:
                    if neutralize is False:
                        order = position_check(instr_bmex, get_ask(0))
                    if neutralize is True:
                        order = position_check(instr_bmex, get_ask(0), neutralize=True)
                if (time_actual > time_cached + sec_to_destroy * 1000 or counter >= quotes_to_destroy) and neutralize is False:
                    logger.info(instr_bmex + ': Initial Timer Failed !')
                    if paper_trading is False:
                        client.Order.Order_cancelAll().result()
                        # launch_order(definition='market', direction='buy', size=size_official)
                        # logger.info('Buy market order filled !')
                        return True
                    else:
                        # order = 0
                        os.remove(naming_order)
                        # position_check(get_ask(0), market=True)
                        if neutralize is True:
                            return True
                        try:
                            open('pnl.csv')
                            df = pd.read_csv(r'pnl.csv')
                            current_pnl = df['pnl'].iloc[0]
                            # logger.info('Buy market order filled !')
                            if current_pnl >= 0:
                                logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTGREEN_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
                            if current_pnl < 0:
                                logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTRED_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
                            return True
                        except Exception as e:
                            # logger.info('Buy market order filled !')
                            return True
                if bid_cached < matrix_bmex_ticker[2]:
                    counter += 1
                    if neutralize is True:
                        if paper_trading is False:
                            client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[2]).result()
                        else:
                            df = pd.read_csv(naming_order)
                            open_price = df['price'].iloc[0]
                            if open_price > get_bid(0):
                                launch_paper_order(instr_bmex, direction='long', size=size_official, price=get_bid(1))
                            else:
                                launch_paper_order(instr_bmex, direction='long', size=size_official, price=get_bid(0))
                    else:
                        if paper_trading is False:
                            client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[2]).result()
                        if paper_trading is True:
                            launch_paper_order(instr_bmex, direction='long', size=size_official, price=matrix_bmex_ticker[2])
                    bid_cached = matrix_bmex_ticker[2]
                    logger.info(instr_bmex + ": BUY LIMIT moved @" + str(matrix_bmex_ticker[2]))
                    # sleep(0.5)
                    continue
                # time_cached = get_time(instr_bmex)
                sleep(0.5)
                continue
        except IndexError:
            if paper_trading is False and \
                    ws_bmex.open_orders_raw(instr_bmex)[len(ws_bmex.open_orders_raw(instr_bmex)) - 1]['ordStatus'] != "Filled":
                # launch_order(definition='market', direction='buy', size=size_official)
                client.Order.Order_cancelAll().result()  # Clear remaining stops
                # logger.info('Emergency buy market order filled !')
                return True
            pass
        except HTTPBadRequest as e:
            logger.error(str(e))
            logger.info('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
            sleep(1)
            pass
        except HTTPServiceUnavailable:
            logger.exception('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
            sleep(1)
            pass
        if paper_trading is True:
            try:
                open('pnl.csv')
                df = pd.read_csv(r'pnl.csv')
                current_pnl = df['pnl'].iloc[0]
                logger.info(instr_bmex + ': Buy order filled !')
                if current_pnl >= 0:
                    logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTGREEN_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
                if current_pnl < 0:
                    logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTRED_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
            except Exception as e:
                logger.info(instr_bmex + ': Buy order filled !')
                return True
        return True
    except HTTPBadRequest as e:
        logger.error(str(e))
        logger.info('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
        sleep(1)
        pass
    except HTTPServiceUnavailable:
        logger.exception('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
        sleep(1)
        pass


def fire_sell(instr_bmex, pos_size, contrarian_=False, neutralize=False):

    naming_order = str(instr_bmex) + '_order.csv'
    naming_pos = str(instr_bmex) + '_position.csv'

    try:
        matrix_bmex_ticker = [None] * 3
        if neutralize is False:
            logger.info(instr_bmex + ": --- Initiating SELL strategy ---")
        else:
            if paper_trading is True:
                try:
                    open(naming_order)
                    os.remove(naming_order)
                except Exception as e:
                    try:
                        open(naming_pos)
                        pass
                    except Exception as e:
                        logger.info(instr_bmex + ": --- Stopping Neutralizing Mode ---")
                        return True
                logger.info(instr_bmex + ": --- Initiating Neutralizing Mode ---")
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
        else:
            try:
                open(naming_pos)
                df = pd.read_csv(naming_pos)
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
                pass
        if paper_trading is False:
            sl_ord_number = launch_order(instr_bmex, definition='limit', direction='sell', size=size_official, price=matrix_bmex_ticker[1])
        if paper_trading is True:
            launch_paper_order(instr_bmex, direction='short', size=size_official, price=matrix_bmex_ticker[1])
            order = 1
        counter = 0
        if paper_trading is False:
            try:
                while ws_bmex.open_orders_raw(instr_bmex) == [] or ws_bmex.open_orders_raw(instr_bmex)[len(ws_bmex.open_orders_raw(instr_bmex)) - 1]['ordStatus'] != "New":
                    sleep(0.1)
                    counter += 1
                    if counter >= 100:
                        break
            except Exception as e:
                logger.exception(e)
                pass
        ask_cached = matrix_bmex_ticker[1]
        time_cached = get_time()
        counter = 0
        try:
            while (order == 1 and paper_trading is True) \
                    or (ws_bmex.open_orders_raw(instr_bmex) != [] and ws_bmex.open_orders_raw(instr_bmex)[len(ws_bmex.open_orders_raw(instr_bmex)) - 1]['ordStatus'] == "New" and paper_trading is False):
                matrix_bmex_ticker[1] = get_ask(spread - 1)
                matrix_bmex_ticker[2] = get_bid(spread - 1)

                time_actual = get_time()
                if ((learning.get_p_verdict() == 1 and contrarian_ is False) or (learning.get_p_verdict() < 0 and contrarian_ is True)) and neutralize is False:
                    if paper_trading is False:
                        client.Order.Order_cancelAll().result()
                    else:
                        # order = 0
                        os.remove(naming_order)
                    logger.info(instr_bmex + ': Initial bearish micro-trend just reversed !')
                    return False
                if paper_trading is True:
                    if neutralize is False:
                        order = position_check(instr_bmex, get_bid(0))
                    if neutralize is True:
                        order = position_check(instr_bmex, get_bid(0), neutralize=True)
                if (time_actual > time_cached + sec_to_destroy * 1000 or counter >= quotes_to_destroy) and neutralize is False:
                    logger.info(instr_bmex + ': Initial Timer Failed !')
                    if paper_trading is False:
                        client.Order.Order_cancelAll().result()
                        # launch_order(definition='market', direction='sell', size=size_official)
                        # logger.info('Sell market order filled !')
                        return True
                    else:
                        # order = 0
                        os.remove(naming_order)
                        # position_check(get_bid(0), market=True)
                        if neutralize is True:
                            return True
                        try:
                            open('pnl.csv')
                            df = pd.read_csv(r'pnl.csv')
                            current_pnl = df['pnl'].iloc[0]
                            # logger.info('Sell market order filled !')
                            if current_pnl >= 0:
                                logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTGREEN_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
                            if current_pnl < 0:
                                logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTRED_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
                            return True
                        except Exception as e:
                            # logger.exception('Sell market order filled !')
                            return True
                if ask_cached > matrix_bmex_ticker[1]:
                    counter += 1
                    if neutralize is True:
                        if paper_trading is False:
                            client.Order.Order_amend(orderID=sl_ord_number, price=matrix_bmex_ticker[1]).result()
                        else:
                            df = pd.read_csv(naming_order)
                            open_price = df['price'].iloc[0]
                            if open_price < get_ask(0):
                                launch_paper_order(instr_bmex, direction='short', size=size_official,
                                                        price=get_ask(1))
                            else:
                                launch_paper_order(instr_bmex, direction='short', size=size_official,
                                                        price=get_ask(0))
                    else:
                        if paper_trading is False:
                            client.Order.Order_amend(orderID=sl_ord_number,
                                                              price=matrix_bmex_ticker[1]).result()
                        if paper_trading is True:
                            launch_paper_order(instr_bmex, direction='short', size=size_official,
                                                    price=matrix_bmex_ticker[1])
                    ask_cached = matrix_bmex_ticker[1]
                    logger.info(instr_bmex + ": SELL LIMIT moved @" + str(matrix_bmex_ticker[1]))
                    continue
                # time_cached = get_time(instr_bmex)
                sleep(0.5)
                continue
        except IndexError:
            if paper_trading is False and \
                    ws_bmex.open_orders_raw(instr_bmex)[len(ws_bmex.open_orders_raw(instr_bmex)) - 1][
                        'ordStatus'] != "Filled":
                # launch_order(definition='market', direction='sell', size=size_official)
                client.Order.Order_cancelAll().result()  # Clear remaining stops
                # logger.info('Emergency sell market order filled !')
                return True
            pass
        except HTTPBadRequest as e:
            logger.error(str(e))
            logger.info('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
            sleep(1)
            pass
        except HTTPServiceUnavailable:
            logger.exception('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
            sleep(1)
            pass
        if paper_trading is True:
            try:
                open('pnl.csv')
                df = pd.read_csv(r'pnl.csv')
                current_pnl = df['pnl'].iloc[0]
                logger.info(instr_bmex + ': Sell order filled !')
                if current_pnl >= 0:
                    logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTGREEN_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
                if current_pnl < 0:
                    logger.info(instr_bmex + ': Realized PnL -> ' + Fore.LIGHTRED_EX + str(round(current_pnl, 2)) + Fore.WHITE + ".")
            except Exception as e:
                logger.info(instr_bmex + ': Sell order filled !')
                return True
        return True
    except HTTPBadRequest as e:
        logger.error(str(e))
        logger.info('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
        sleep(1)
        pass
    except HTTPServiceUnavailable:
        logger.exception('Bravado error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
        sleep(1)
        pass


def main():
    matrix_bmex_ticker = [None] * 4
    odbk_cached = [None]
    hour_cached = None
    counter_trial = 0
    naming_pos = str(instrument_bmex) + '_position.csv'

    logger.info('It began in Africa')

    model_exist = False

    if bb_protect is True:
        bb.start_bb()

    rsi.start_rsi()
    learning.start_ml()

    try:
        while True:
            thr1 = thr2 = 0
            DT = ws_bmex.get_instrument()['timestamp']
            dt2ts = dt.utcnow().timestamp()
            dt2retrieval = dt.strptime(DT, '%Y-%m-%dT%H:%M:%S.%fZ')  # .replace(tzinfo=timezone.utc)
            matrix_bmex_ticker[0] = int(dt2ts * 1000)
            matrix_bmex_ticker[1] = get_ask_size(0)
            matrix_bmex_ticker[2] = get_bid_size(0)
            matrix_bmex_ticker[3] = get_ask_size(0) + get_bid_size(0)
            hour_actual = dt2retrieval.hour
            if odbk_cached != matrix_bmex_ticker[3]:
                if (hour_actual == time_to_train and hour_cached == time_to_train - 1) \
                        or model_exist is False:
                    logger.info('Starting AutoTraining Module...')
                    if skip_initial_training is False:
                        thr1, thr2 = AutoTrainModel(model_file, instrument_bmex).start()
                    else:
                        naming_thr = str(instrument_bmex) + '_thresholds.csv'
                        try:
                            open(naming_thr)
                            thr_raw = pd.read_csv(naming_thr)
                            thr1 = thr_raw['0'][0]
                            thr2 = thr_raw['0'][1]
                            logger.info("model thr_1: " + str(thr1) + " / model thr_2: " + str(thr2))
                        except Exception as e:
                            logger.error(str(e))
                            return

                    logger.info("Training complete !")
                    if model_exist is False:
                        annihilator = Annihilator(ws=ws_bmex, instrument=instrument_bmex, model_file=model_file, thr_1=thr1, thr_2=thr2)
                    if annihilator.get_status() is False:
                        annihilator.start_annihilator()
                    else:
                        annihilator.stop_annihilator()
                        annihilator = Annihilator(ws=ws_bmex, instrument=instrument_bmex, model_file=model_file, thr_1=thr1, thr_2=thr2)
                        annihilator.start_annihilator()
                    model_exist = True
                hour_cached = hour_actual
                # sleep(0.1)
                while annihilator.get_status() is False or rsi.get_status() is False:
                    sleep(0.1)
                    continue
                verdict = annihilator.get_verdict()
                rsi_value = rsi.get_rsi_value()
                ml_verdict = learning.get_p_verdict()
                if bb_protect is True:
                    bb_verdict = bb.get_verdict()
                else:
                    bb_verdict = 0

                if ((verdict <= -0.5 and rsi_value < rsi_thr_downer) or verdict == -1) and ml_verdict > 0 and bb_verdict != 1:
                    if paper_trading is False:
                        if abs(ws_bmex.open_positions()) < max_pos or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() < 0):
                            if contrarian is False:
                                logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                buy_action = fire_buy(instrument_bmex, pos_size)
                            elif abs(ws_bmex.open_positions()) < max_pos:
                                logger.info('CONTRARIAN SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                buy_action = fire_sell(instrument_bmex, pos_size, contrarian_=True)
                            if buy_action is True:
                                if instrument_bmex[-3:] == 'USD':
                                    logger.info('Balance: ' + str(client.User.User_getWallet(currency='XBt').result()[0]['amount']))
                                elif instrument_bmex[-4:] == 'USDT':
                                    logger.info('Balance: ' + str(client.User.User_getWallet(currency='USDt').result()[0]['amount']))
                    if paper_trading is True:
                        try:
                            open(naming_pos)
                            df = pd.read_csv(naming_pos)
                            current_size = df['size'].iloc[0]
                            if abs(current_size) < max_pos or (abs(current_size) >= max_pos and current_size < 0):
                                if contrarian is False:
                                    logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                    fire_buy(instrument_bmex, pos_size)
                                elif abs(current_size) < max_pos:
                                    logger.info('CONTRARIAN SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                    fire_sell(instrument_bmex, pos_size, contrarian_=True)
                        except Exception as e:
                            if contrarian is False:
                                logger.info('BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                    round(verdict, 3)))
                                fire_buy(instrument_bmex, pos_size)
                            else:
                                logger.info(
                                    'CONTRARIAN SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                        round(verdict, 3)))
                                fire_sell(instrument_bmex, pos_size, contrarian_=True)

                if ((verdict >= 0.5 and rsi_value > rsi_thr_upper) or verdict == 1) and ml_verdict < 0 and bb_verdict != -1:
                    if paper_trading is False:
                        if abs(ws_bmex.open_positions()) < max_pos \
                                or (abs(ws_bmex.open_positions()) >= max_pos and ws_bmex.open_positions() > 0):
                            if contrarian is False:
                                logger.info(
                                    'SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(round(verdict, 3)))
                                sell_action = fire_sell(instrument_bmex, pos_size)
                            elif abs(ws_bmex.open_positions()) < max_pos:
                                logger.info('CONTRARIAN BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                    round(verdict, 3)))
                                sell_action = fire_buy(instrument_bmex, pos_size, contrarian_=True)
                            if sell_action is True:
                                if instrument_bmex[-3:] == 'USD':
                                    logger.info('Balance: ' + str(client.User.User_getWallet(currency='XBt').result()[0]['amount']))
                                elif instrument_bmex[-4:] == 'USDT':
                                    logger.info('Balance: ' + str(client.User.User_getWallet(currency='USDt').result()[0]['amount']))
                    if paper_trading is True:
                        try:
                            open(naming_pos)
                            df = pd.read_csv(naming_pos)
                            current_size = df['size'].iloc[0]
                            if abs(current_size) < max_pos \
                                    or (abs(current_size) >= max_pos and current_size > 0):
                                if contrarian is False:
                                    logger.info('SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                        round(verdict, 3)))
                                    fire_sell(instrument_bmex, pos_size)
                                elif abs(current_size) < max_pos:
                                    logger.info(
                                        'CONTRARIAN BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                            round(verdict, 3)))
                                    fire_buy(instrument_bmex, pos_size, contrarian_=True)
                        except Exception as e:
                            if contrarian is False:
                                logger.info('SELL ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                    round(verdict, 3)))
                                fire_sell(instrument_bmex, pos_size)
                            else:
                                logger.info(
                                    'CONTRARIAN BUY ! RSI: ' + str(round(rsi_value, 2)) + ' - Verdict: ' + str(
                                        round(verdict, 3)))
                                fire_buy(instrument_bmex, pos_size, contrarian_=True)

                if paper_trading is False and (bb_verdict == 1 or bb_verdict == -1) and ws_bmex.open_positions() != 0:
                    logger.info('Neutralization Starting')
                    if ws_bmex.open_positions() > 0:
                        neutralization = fire_sell(instrument_bmex, pos_size, neutralize=True)
                    else:
                        neutralization = fire_buy(instrument_bmex, pos_size, neutralize=True)
                    if neutralization is True:
                        logger.info('Neutralization Over, standing by...')
                    else:
                        counter_trial += 1
                    if counter_trial > 5:
                        if ws_bmex.open_positions() > 0:
                            launch_order(instrument_bmex, definition='market', direction='sell', size=abs(ws_bmex.open_positions()))
                        if ws_bmex.open_positions() < 0:
                            launch_order(instrument_bmex, definition='market', direction='buy', size=abs(ws_bmex.open_positions()))
                        counter_trial = 0
                        logger.info('Forced Closure, standing by...')
                if paper_trading is True and (bb_verdict == 1 or bb_verdict == -1):
                    try:
                        open(naming_pos)
                        df = pd.read_csv(naming_pos)
                        current_size = df['size'].iloc[0]
                        if current_size != 0:
                            logger.info('Neutralization Starting')
                            if current_size > 0:
                                neutralization = fire_sell(instrument_bmex, pos_size, neutralize=True)
                            else:
                                neutralization = fire_buy(instrument_bmex, pos_size, neutralize=True)
                            if neutralization is True:
                                logger.info('Neutralization Over, standing by...')
                                os.remove(naming_pos)
                    except Exception as e:
                        pass

                odbk_cached = matrix_bmex_ticker[3]
            sleep(0.005)

    except HTTPServerError:
        logger.error('WS error: ' + Fore.LIGHTCYAN_EX + 'Resetting' + Fore.WHITE + '...')
        sleep(1)
        pass

    except Exception as e:
        logger.exception(str(e))
        sleep(1)
        if len(ws_bmex.open_stops()) != 0:
            client.Order.Order_cancelAll().result()
        sleep(1)
        pass
        # raise


if __name__ == '__main__':
    signal(SIGINT, handler)
    main()
