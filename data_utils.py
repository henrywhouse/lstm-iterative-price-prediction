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

########## PRE-PROCESSING ##########

def arr_to_windowed_xy(arr, look_back):
    
    x = []
    y = []

    for i in range(look_back,arr.shape[0]):
        x.append(arr[i-look_back:i])
        y.append(arr[i][-1])

    return  np.array(x), np.array(y)

def mm_scale(df):
    dates = np.array(df.index)
    mm_sc = MinMaxScaler()
    data_sc = mm_sc.fit_transform(df)
    return dates, data_sc

def pca(df, n_components):
    principal_components=PCA(n_components=n_components)
    principal_components.fit(df)
    pc=principal_components.transform(df)
    pc_df = pd.DataFrame()
    for i in range(n_components):
        pc_df[f'PC {i+1}'] = pc[:,i]
    pc_df.index = df.index
    return pc_df

########## POST-PREDICTION ANALYSIS ##########

def simple_backtest(look_back, dates_test, y_test, test_pred, short_sell=False, comp_prev=False, show_plot=False):
    df = pd.DataFrame({'Obs':y_test, 'Pred':test_pred})
    df.index = dates_test[look_back:]
    if comp_prev:
        df['Strat'] = df['Pred'] > df['Pred'].shift(1)
    else:
        df['Strat'] = df['Pred'] > df['Obs'].shift(1)
    df['Strat'] = df['Strat'].dropna().astype(int)
    if short_sell:
        df['Strat'] = df['Strat'].replace(0, -1)
    df['Long Daily Ret'] = df['Obs'].pct_change()
    df['Strat Daily Ret'] = df['Strat']*df['Obs'].pct_change()
    df['Long Cum Ret'] = (df['Long Daily Ret']+1).cumprod() - 1
    df['Strat Cum Ret'] = (df['Strat Daily Ret']+1).cumprod() - 1
    if show_plot:
        plot_backtest(df)
    return df

########## PLOTS ##########

def plot_predictions(df_dict):
    names = ['Train', 'Test'] # Add Validation Later
    fig = make_subplots(rows=2, cols=1)
    for k in range(2):
        fig.add_trace(
            go.Scatter(x=df_dict[names[k]].index,
                        y=df_dict[names[k]]['Actual'],
                        name=f'{names[k]} Actual'),
                        row=k+1, col=1 )
        fig.add_trace(
            go.Scatter(x=df_dict[names[k]].index,
                        y=df_dict[names[k]]['Predicted'],
                        name=f'{names[k]} Predicted'),
                        row=k+1, col=1)
        fig.update_layout(height=600, width=800, 
                          title_text="Predictions for Train, Test Sets")
    fig.show()

def plot_backtest(df):
    fig = px.line(df, x=df.index, y=['Long Cum Ret', 'Strat Cum Ret'])
    fig.update_layout({"title": 'Backtest Results',
                        "xaxis": {"title":"Date"},
                        "yaxis": {"title":"Cumulative Return"},
                        "showlegend": True})
    fig.show()