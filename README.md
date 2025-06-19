# AlphaFolio YFinance API

A unified FastAPI backend server for fetching current and historical stock market data using Yahoo Finance.

## Features

- **Five comprehensive endpoints**: `/current`, `/historical`, `/info`, `/dividends`, `/splits`
- **Multiple ticker support**: POST method for batch stock queries
- **Error handling**: Graceful error handling per ticker
- **Real-time data**: Current stock prices, changes, volume, and market data
- **Historical data**: Configurable periods and intervals
- **Company information**: Comprehensive fundamental data and metrics
- **Dividend analysis**: Complete dividend history and calculations
- **Stock splits tracking**: Historical splits with cumulative factors

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Current Stock Data - `/current`

Get current stock prices and market data.

#### Multiple Tickers (POST)

```bash
# Post multiple tickers as JSON
curl -X POST "http://localhost:8000/current" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL", "TSLA"]}'
```

**Response Format:**

```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "current_price": 150.25,
      "previous_close": 148.5,
      "market_cap": 2500000000000,
      "volume": 45000000,
      "day_high": 151.0,
      "day_low": 149.0,
      "change": 1.75,
      "change_percent": 1.18,
      "currency": "USD",
      "error": null
    }
  ],
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 2. Historical Stock Data - `/historical`

Get historical stock data with configurable periods and intervals.

#### Multiple Tickers (POST)

```bash
# Post multiple tickers with parameters
curl -X POST "http://localhost:8000/historical?period=3mo&interval=1d" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'
```

**Response Format:**

```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "data": [
        {
          "date": "2024-01-10",
          "open": 148.0,
          "high": 151.2,
          "low": 147.5,
          "close": 150.25,
          "volume": 45000000
        }
      ],
      "error": null
    }
  ],
  "period": "1mo",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 3. Company Information - `/info`

Get comprehensive company information and fundamental data.

#### Multiple Tickers (POST)

```bash
# Get company information for multiple stocks
curl -X POST "http://localhost:8000/info" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'
```

**Response Format:**

```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "market_cap": 2500000000000,
      "enterprise_value": 2480000000000,
      "pe_ratio": 25.5,
      "forward_pe": 22.8,
      "peg_ratio": 1.2,
      "price_to_book": 8.9,
      "price_to_sales": 6.1,
      "beta": 1.2,
      "dividend_rate": 0.96,
      "dividend_yield": 0.0048,
      "payout_ratio": 0.15,
      "fifty_two_week_high": 198.23,
      "fifty_two_week_low": 164.08,
      "shares_outstanding": 15204100000,
      "float_shares": 15204100000,
      "employees": 164000,
      "currency": "USD",
      "country": "United States",
      "website": "https://www.apple.com",
      "business_summary": "Apple Inc. designs, manufactures, and markets smartphones...",
      "error": null
    }
  ],
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 4. Dividend History - `/dividends`

Get complete dividend payment history and analysis.

#### Multiple Tickers (POST)

```bash
# Get dividend history for multiple stocks
curl -X POST "http://localhost:8000/dividends" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "KO"]}'
```

**Response Format:**

```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "dividends": [
        {
          "date": "2024-02-16",
          "dividend": 0.24
        },
        {
          "date": "2023-11-16",
          "dividend": 0.23
        }
      ],
      "total_dividends": 45.67,
      "dividend_count": 120,
      "annual_dividend_rate": 0.96,
      "error": null
    }
  ],
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 5. Stock Splits History - `/splits`

Get complete stock split history with cumulative factors.

#### Multiple Tickers (POST)

```bash
# Get stock splits for multiple stocks
curl -X POST "http://localhost:8000/splits" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "TSLA", "AMZN"]}'
```

**Response Format:**

```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "splits": [
        {
          "date": "2020-08-31",
          "split_ratio": 4.0
        },
        {
          "date": "2014-06-09",
          "split_ratio": 7.0
        }
      ],
      "total_splits": 5,
      "cumulative_split_factor": 224.0,
      "error": null
    }
  ],
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

## Parameters

### Historical Data Parameters

- **period**: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`
- **interval**: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

## Usage Examples

### JavaScript/Node.js

```javascript
// Current data for multiple tickers (POST)
const response = await fetch("http://localhost:8000/current", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ tickers: ["AAPL", "MSFT", "GOOGL"] }),
});

// Historical data for multiple tickers (POST)
const response = await fetch("http://localhost:8000/historical?period=1mo&interval=1d", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ tickers: ["AAPL", "MSFT", "GOOGL"] }),
});

// Company info for multiple tickers (POST)
const response = await fetch("http://localhost:8000/info", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ tickers: ["AAPL", "MSFT", "GOOGL"] }),
});

// Dividends for multiple tickers (POST)
const response = await fetch("http://localhost:8000/dividends", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ tickers: ["AAPL", "MSFT", "KO"] }),
});

// Stock splits for multiple tickers (POST)
const response = await fetch("http://localhost:8000/splits", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ tickers: ["AAPL", "TSLA", "AMZN"] }),
});
```

### Python

```python
import requests

# Current data for multiple tickers (POST)
response = requests.post('http://localhost:8000/current',
                        json={'tickers': ['AAPL', 'MSFT', 'GOOGL']})

# Historical data for multiple tickers (POST)
response = requests.post('http://localhost:8000/historical',
                        json={'tickers': ['AAPL', 'MSFT', 'GOOGL']},
                        params={'period': '1mo', 'interval': '1d'})

# Company info for multiple tickers (POST)
response = requests.post('http://localhost:8000/info',
                        json={'tickers': ['AAPL', 'MSFT', 'GOOGL']})

# Dividends for multiple tickers (POST)
response = requests.post('http://localhost:8000/dividends',
                        json={'tickers': ['AAPL', 'MSFT', 'KO']})

# Stock splits for multiple tickers (POST)
response = requests.post('http://localhost:8000/splits',
                        json={'tickers': ['AAPL', 'TSLA', 'AMZN']})
```

## Testing

Run the test suite to verify all endpoints:

```bash
python test_api.py
```

## Error Handling

- Individual tickers that fail will have an `error` field in their response
- Invalid parameters return HTTP 400 with error details
- Server errors return HTTP 500 with error details
- The API gracefully handles invalid tickers and network issues

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger/OpenAPI.

## Health Check

Check if the server is running:

```bash
curl http://localhost:8000/health
```
