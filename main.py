import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
from bs4 import BeautifulSoup

# Part 1: NIFTY50 Stock Price Prediction

# 1. Web Scraping

def scrape_nifty50_data(url):
    """
    Scrapes NIFTY50 historical stock data from a given URL.

    Args:
        url: The URL of the webpage containing NIFTY50 data.

    Returns:
        A pandas DataFrame containing the scraped data.
    """

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'economic-calendar-table'})

    # Extract data from the table
    data = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        date = cells[0].text.strip()
        close = float(cells[4].text.replace(',', ''))
        data.append([date, close])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = df.set_index('Date')
    return df

# Scrape data for the last six months
start_date = pd.Timestamp.today() - pd.DateOffset(months=6)
end_date = pd.Timestamp.today()
url = f"https://www.moneycontrol.com/india/stockpricequote/index/nifty50/{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}/"
nifty50_data = scrape_nifty50_data(url)

# 2. Linear Regression Model

# Split data into training and testing sets
train_data = nifty50_data[:-30]
test_data = nifty50_data[-30:]

# Prepare data for training
X_train = train_data.index.values.reshape(-1, 1)
y_train = train_data['Close'].values

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Prediction and Evaluation

# Predict on test data
X_test = test_data.index.values.reshape(-1, 1)
y_pred = model.predict(X_test)

# Calculate root mean squared error
rmse = mean_squared_error(test_data['Close'], y_pred, squared=False)

# Visualize actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Close'], label='Actual')
plt.plot(test_data.index, y_pred, label='Predicted')
plt.title('NIFTY50 Stock Price Prediction (Last Month)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

print(f"Root Mean Squared Error: {rmse:.2f}")