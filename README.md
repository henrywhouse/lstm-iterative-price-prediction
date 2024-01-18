# LSTM Iterative Price Prediction

## Use customized LSTM neural networks trained on historical stock prices and macroeconomic indicators to construct efficient, alpha-generating portfolios 

### Introduction
This repository is a data science project that attempts to construct a portfolio that outperforms its benchmark using neural networks with an LSTM layer in its architecture. The networks are given a matrix of data of size (look_back, n_features)--a sliding window of the previous look_back day's data--and seek to predict the next day's close price of the stock. The features in this implementation are the adjusted close, volume, and two macroeconomic indicators: the VIX and the 3-month treasury rate. The length of the dataset is July 7, 2009, to March 1, 2023. After prediction, we create a strategy in which if the neural network predicts an increase in the price we buy, and if it predicts a decline, we sell. The portfolio is weighted according to the prorated weights of the S&P 500 (e.g., if we take the top 20 stocks, we re-weight those so they add up to 1).

### Pre-Processing
To pre-process the data, we begin by applying a MinMax() scaler to each of the matrices of stock prices to create a boundary of (0,1). This is particularly important when including volume data as that can be many orders of magnitude greater than our price or macroeconomic data which can adversely affect the predictive power of the networks. Next, we concatenate all features of interest and split the data into 3 sets: train, validation, and test representing in this example 80%, 10%, and 10%, respectively of the overall dataset. We then create our sliding window of data with a period of look_back, yielding our observation size of (look_back, n_features).

### Network architecture
For the neural networks, we use a simple architecture consisting of an LSTM layer and 3 Dense layers. The LSTM layer in this instance has 256 units and a tanh activation. We use tanh activation for a dual purpose: it showed the lowest average error on our validations et and also permits GPU acceleration for training the networks. The next two Dense layers have 512 and 256 units respectively and ReLU activation functions. The output layer is Dense with 1 unit.

### Training & Prediction 
For each asset in our list of stocks, we train a unique neural network of the architecture specified above. Hyperparameter tuning was done using validation mean squared error (MSE) and mean absolute error (MAE). We save all predictions on the entirety of the test set.

### Post-Processing
For each observation in our test set (November 16, 2022, to February 28, 2023), we have a simple method that calculates our position for the next day given our networks' predictions. As mentioned above, if it predicts higher prices than today's actual price, we go long, and if it predicts lower, we sell our holding. We take these positions and calculate a strategy return for each indiviudal stock and then blend the individual returns weighted by market cap to get a portfolio. 

### Portfolio Statistics
To analyze the overall performance of our algorithm, we analyze multiple common statistics for portfolio returns, volume, and efficiency.

### Limitations


### Future Model Development
