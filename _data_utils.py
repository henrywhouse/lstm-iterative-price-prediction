import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import json
import config
import requests
from bs4 import BeautifulSoup as bs
from fredapi import Fred
from config import fred_api_key

########## DATA IMPORTS ##########

def get_sp_ticker_list():
    url = 'https://www.slickcharts.com/sp500'
    headers = config.headers
    response = requests.get(url, headers=headers)
    soup = bs(response.content,'html.parser')
    tables = soup.find_all('table','table')
    s_p_df = pd.read_html(str(tables[0]))[0]
    s_p_df['Symbol'] = s_p_df['Symbol'].str.replace('.','-')
    s_p_df['Weight'] = s_p_df['Portfolio%'].apply(lambda x: np.float32(x.rstrip('%')))
    s_p_df.drop(columns=['Portfolio%'], inplace=True)
    return s_p_df

def get_stock_prices(tickers, start_date, end_date, interval='1d'):
    data_dict = {}
    for ticker in tickers:
        data_dict[ticker] = yf.download(ticker, start_date, end_date, interval=interval)
    return data_dict

def get_fred_data(fred, series, start_date, end_date):
    df = pd.DataFrame()
    series_dict = series
    for key in series_dict:
        series = fred.get_series(key)
        series = series.replace('.', 'NaN')
        df[series_dict[key]] = series
    df = df[start_date:end_date]
    return df

def init_fred():
    fred = Fred(api_key=fred_api_key)
    return fred

