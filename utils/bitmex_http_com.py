try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import time, hashlib, hmac
import pandas as pd
import threading
from time import sleep
from datetime import datetime as dt, timedelta
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient, Authenticator
from bravado.swagger_model import Loader

# swagger spec's formats to exclude. this help to avoid warning in your console.
EXCLUDE_SWG_FORMATS = ['JSON', 'guid']


class BollingerCalculus:

    def __init__(self, instrument, period, test, api_key, api_secret):
        self.instrument = instrument
        self.server_time = dt.utcnow()
        self.period = period
        self.test = test
        self.api_key = api_key
        self.api_secret = api_secret

        self.df = pd.DataFrame(columns=['Date', 'Close'])
        self.verdict = 0

    def retrieve_data(self, server_time, k):
        time_starter = server_time + timedelta(seconds=1) - timedelta(minutes=k)
        # print(time_starter)
        client = bitmex(test=self.test, api_key=self.api_key, api_secret=self.api_secret)
        self.df = pd.DataFrame(columns=['Date', 'Close'])
        while len(self.df) < k:
            self.df = pd.DataFrame(columns=['Date', 'Close'])
            data = client.Trade.Trade_getBucketed(symbol=self.instrument, binSize="1m", count=k,
                                                  startTime=time_starter).result()
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
        server_minute_cached = None
        while True:
            actual_time = dt.utcnow()
            dt2ts = dt.utcnow().timestamp()
            dt2retrieval = actual_time
            server_minute = int(dt2ts // 60 % 60)
            if server_minute_cached != server_minute:
                server_minute_cached = server_minute
                print('Updating BB')
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
                    print('BB: Excess to the upside !')
                    self.verdict = 1
                elif last_data.iloc[0].at['Close'] < last_data.iloc[0].at['Lower_Band']:
                    print('BB: Excess to the downside !')
                    self.verdict = -1
                elif last_data.iloc[0].at['Close'] > last_data.iloc[0].at['Lower_Band'] and last_data.iloc[0].at['Close'] < last_data.iloc[0].at['Lower_Band_']:
                    print('BB: No sell allowed !')
                    self.verdict = -0.5
                elif last_data.iloc[0].at['Close'] < last_data.iloc[0].at['Upper_Band'] and last_data.iloc[0].at['Close'] > last_data.iloc[0].at['Upper_Band_']:
                    print('BB: No buy allowed !')
                    self.verdict = 0.5
                else:
                    print('Close: ' + str(last_data.iloc[0].at['Close'])
                          + ' / Upper: ' + str(last_data.iloc[0].at['Upper_Band'])
                          + ' / Downer: ' + str(last_data.iloc[0].at['Lower_Band']))
                    self.verdict = 0

    def start_bb(self):
        thread = threading.Thread(target=self.compute_bb)
        thread.start()

    def get_verdict(self):
        return self.verdict



class APIKeyAuthenticator(Authenticator):

    def __init__(self, host, api_key, api_secret):
        super(APIKeyAuthenticator, self).__init__(host)
        self.api_key = api_key
        self.api_secret = api_secret

    def matches(self, url):
        if 'swagger.json' in url:
            return False
        return True

    def apply(self, r):
        # 5s grace period in case of clock skew
        expires = int(time.time() * 1000)
        r.headers['api-expires'] = str(expires)
        r.headers['api-key'] = self.api_key
        prepared = r.prepare()
        body = prepared.body or ''
        url = prepared.path_url
        r.headers['api-signature'] = self.generate_signature(self.api_secret, r.method, url, expires, body)
        return r

    def generate_signature(self, secret, verb, url, nonce, data):
        parsedURL = urlparse(url)
        path = parsedURL.path
        if parsedURL.query:
            path = path + '?' + parsedURL.query

        nonce = str(nonce)
        _message = verb + path + nonce + data

        message = bytes(_message.encode('utf-8'))
        secret = bytes(secret.encode('utf-8'))

        return hmac.new(secret, message, digestmod=hashlib.sha256).hexdigest()


def bitmex(test=True, config=None, api_key=None, api_secret=None):
    # config options at http://bravado.readthedocs.io/en/latest/configuration.html
    if not config:
        config = {
            # Don't use models (Python classes) instead of dicts for #/definitions/{models}
            'use_models': False,
            # bravado has some issues with nullable fields
            'validate_responses': False,
            # Returns response in 2-tuple of (body, response); if False, will only return body
            'also_return_response': True,

            # 'validate_swagger_spec': True,
            # 'validate_requests': True,
            # 'formats': [],
        }

    host = 'https://www.bitmex.com'
    if test:
        host = 'https://testnet.bitmex.com'

    spec_uri = host + '/api/explorer/swagger.json'
    spec_dict = get_swagger_json(spec_uri, exclude_formats=EXCLUDE_SWG_FORMATS)

    if api_key and api_secret:
        request_client = RequestsClient()
        request_client.authenticator = APIKeyAuthenticator(host, api_key, api_secret)
        return SwaggerClient.from_spec(spec_dict, origin_url=spec_uri, http_client=request_client, config=config)
    else:
        return SwaggerClient.from_spec(spec_dict, origin_url=spec_uri, http_client=None, config=config)


# exclude some format from swagger json to avoid warning in API execution.
def get_swagger_json(spec_uri, exclude_formats=[]):
    loader = Loader(RequestsClient())
    spec_dict = loader.load_spec(spec_uri)
    if not exclude_formats:
        return spec_dict

    # exlude formats from definitions
    for def_key, def_item in spec_dict['definitions'].items():
        if 'properties' not in def_item:
            continue
        for prop_key, prop_item in def_item['properties'].items():
            if 'format' in prop_item and prop_item['format'] in exclude_formats:
                prop_item.pop('format')

    # exlude formats from paths
    for path_key, path_item in spec_dict['paths'].items():
        for method_key, method_item in path_item.items():
            if 'parameters' not in method_item:
                continue
            for param in method_item['parameters']:
                if 'format' in param and param['format'] in exclude_formats:
                    param.pop('format')

    return spec_dict
