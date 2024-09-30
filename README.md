NIFTY50 Stock Price Prediction

This project involves scraping the historical stock price data of NIFTY50, a benchmark index of the Indian stock market, and using a Linear Regression model to predict the stock prices for the last month. It also provides a visualization of actual vs predicted stock prices, along with the calculation of Root Mean Squared Error (RMSE) to evaluate the model's performance.
Table of Contents

    Overview
    Requirements
    Installation
    Usage
    Files
    Model Explanation
    Evaluation
    License

Overview

This project is broken down into several key steps:

    Web Scraping: The historical stock data for the NIFTY50 index is scraped from the web using the BeautifulSoup library.
    Data Processing: The data is cleaned, organized into a Pandas DataFrame, and prepared for model training.
    Model Training: A Linear Regression model is trained using data from the last six months.
    Prediction & Evaluation: The model predicts stock prices for the last 30 days, and the RMSE is calculated to measure prediction accuracy.
    Visualization: A plot is generated showing both actual and predicted stock prices over the test period.

Requirements

To run the code, the following libraries are required:

    pandas
    numpy
    matplotlib
    sklearn
    requests
    BeautifulSoup4

Installation

You can install the required dependencies using pip:

bash

pip install pandas numpy matplotlib scikit-learn requests beautifulsoup4

Usage

    Scrape NIFTY50 Data: The script scrapes NIFTY50 stock prices from a provided URL.
    Train the Linear Regression Model: After preparing the training and test datasets, the model is trained on the last six months of data.
    Predict Future Prices: The model predicts stock prices for the most recent 30 days.
    Evaluate Model: The Root Mean Squared Error (RMSE) is calculated to measure the accuracy of predictions.
    Visualize Results: The actual vs predicted prices are visualized on a line graph.

Example Usage

python

# Scrape NIFTY50 data for the last 6 months
url = "https://www.moneycontrol.com/india/stockpricequote/index/nifty50/"
nifty50_data = scrape_nifty50_data(url)

# Train the Linear Regression model and predict prices for the last 30 days
X_train, y_train = train_data_preparation(nifty50_data)
model = train_model(X_train, y_train)
evaluate_model(model, test_data)

Files

    nifty50_prediction.py: Main script for scraping data, training the model, making predictions, and visualizing the results.
    README.md: This documentation file.

Model Explanation

The Linear Regression model used in this project assumes that stock prices follow a linear trend. The model attempts to find the best-fit line that minimizes the sum of squared residuals between the actual stock prices and the predicted prices.
Steps:

    The historical stock prices are split into training (all but the last 30 days) and test (last 30 days) datasets.
    The Linear Regression model is trained on the Date (as the independent variable X) and the Closing Price (as the dependent variable y).
    The trained model is then used to predict the stock prices for the test period, which is compared to the actual prices for evaluation.

Evaluation

The model's performance is evaluated using Root Mean Squared Error (RMSE), which gives a measure of the difference between actual and predicted stock prices. A lower RMSE indicates better performance.

For instance:

mathematica

Root Mean Squared Error: 5.67

This suggests that on average, the model's predictions deviate from the actual stock prices by 5.67 points.
License

This project is open-source and licensed under the MIT License.