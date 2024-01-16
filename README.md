# LSTM Iterative Price Prediction

### Summary
This repo is a data science project which attempts to construct a portfolio which outperforms its benchmark using neural netowrks with an LSTM netowrk inn their architecture. The networks are given a msatrix of data of size (look_back, n_features) and seek to predict the next days close price of the stock. The features in this implementation are the adjusted close, volume, and two macroeconomic indicators: the VIX and the 3 month treasury rate. After prediction, we create a strategy in which if the nueral netowrk predicts an increase in the price we buy, and if it predicts a decline, we sell. The portfolio is weighted according to the prorated weights of the S&P 500 (e.g., if we take the top 20 stocks, we re-weight those so they add up to 1).
