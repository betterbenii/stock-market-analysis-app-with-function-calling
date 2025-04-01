# Stock Market Analysis with Function Calling

 This is a Streamlit web application that leverages the yfinance API to provide insights into stocks and their prices. The application uses the Llama 3 model on Groq in conjunction with Langchain to call functions based on the user prompt.

## Key Functions

- **get_stock_info(symbol, key)**: This function fetches various information about a given stock symbol. The information can be anything from the company's address to its financial ratios. The 'key' parameter specifies the type of information to fetch.

- **get_historical_price(symbol, start_date, end_date)**: This function fetches the historical stock prices for a given symbol from a specified start date to an end date. The returned data is a DataFrame with the date and closing price of the stock.

- **get_technical_indicators(symbol, period='6mo')**: This function calculates basic technical indicators for a stock including:
  - SMA (Simple Moving Average) for 20 and 50 days
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)

- **plot_price_over_time(historical_price_dfs)**: This function takes a list of DataFrames (each containing historical price data for a stock) and plots the prices over time using Plotly. The plot is displayed in the Streamlit app.

## Function Calling

The function calling in this application is handled by the Groq API, abstracted with Langchain. When the user asks a question, the application invokes the appropriate tool with parameters based on the user's question. The tool's output is then used to generate a response.

