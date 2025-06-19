from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlphaFolio YFinance API",
    description="A unified backend server for fetching current and historical stock market data using Yahoo Finance",
    version="2.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class StockPrice(BaseModel):
    ticker: str
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    currency: Optional[str] = None
    error: Optional[str] = None

class CurrentDataResponse(BaseModel):
    stocks: List[StockPrice]
    timestamp: datetime = Field(default_factory=datetime.now)

class HistoricalDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class HistoricalStock(BaseModel):
    ticker: str
    data: List[HistoricalDataPoint]
    error: Optional[str] = None

class HistoricalDataResponse(BaseModel):
    stocks: List[HistoricalStock]
    period: str
    timestamp: datetime = Field(default_factory=datetime.now)

class TickersRequest(BaseModel):
    tickers: List[str]

# New models for stock info
class StockInfo(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    beta: Optional[float] = None
    dividend_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    employees: Optional[int] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None
    business_summary: Optional[str] = None
    error: Optional[str] = None

class StockInfoResponse(BaseModel):
    stocks: List[StockInfo]
    timestamp: datetime = Field(default_factory=datetime.now)

# New models for dividends
class DividendData(BaseModel):
    date: str
    dividend: float

class StockDividends(BaseModel):
    ticker: str
    dividends: List[DividendData]
    total_dividends: Optional[float] = None
    dividend_count: Optional[int] = None
    annual_dividend_rate: Optional[float] = None
    error: Optional[str] = None

class DividendsResponse(BaseModel):
    stocks: List[StockDividends]
    timestamp: datetime = Field(default_factory=datetime.now)

# New models for stock splits
class SplitData(BaseModel):
    date: str
    split_ratio: float

class StockSplits(BaseModel):
    ticker: str
    splits: List[SplitData]
    total_splits: Optional[int] = None
    cumulative_split_factor: Optional[float] = None
    error: Optional[str] = None

class SplitsResponse(BaseModel):
    stocks: List[StockSplits]
    timestamp: datetime = Field(default_factory=datetime.now)

# Helper function to get stock info
def get_stock_info(ticker: str) -> StockPrice:
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="2d")
        
        if hist.empty:
            return StockPrice(ticker=ticker, error="No data available")
        
        current_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
        previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
        
        change = None
        change_percent = None
        if current_price and previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
        
        return StockPrice(
            ticker=ticker.upper(),
            current_price=round(current_price, 2) if current_price else None,
            previous_close=round(previous_close, 2) if previous_close else None,
            market_cap=info.get('marketCap'),
            volume=hist['Volume'].iloc[-1] if len(hist) > 0 else None,
            day_high=round(hist['High'].iloc[-1], 2) if len(hist) > 0 else None,
            day_low=round(hist['Low'].iloc[-1], 2) if len(hist) > 0 else None,
            change=round(change, 2) if change else None,
            change_percent=round(change_percent, 2) if change_percent else None,
            currency=info.get('currency', 'USD')
        )
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return StockPrice(ticker=ticker, error=str(e))

# Helper function to get historical data for a single ticker
def get_historical_info(ticker: str, period: str, interval: str) -> HistoricalStock:
    try:
        logger.info(f"Fetching historical data for {ticker} with period {period}")
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return HistoricalStock(ticker=ticker.upper(), data=[], error=f"No historical data found for {ticker}")
        
        data_points = []
        for date, row in hist.iterrows():
            data_points.append(HistoricalDataPoint(
                date=date.strftime('%Y-%m-%d'),
                open=round(row['Open'], 2),
                high=round(row['High'], 2),
                low=round(row['Low'], 2),
                close=round(row['Close'], 2),
                volume=int(row['Volume'])
            ))
        
        return HistoricalStock(
            ticker=ticker.upper(),
            data=data_points
        )
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return HistoricalStock(ticker=ticker.upper(), data=[], error=str(e))

# New helper function to get comprehensive stock info
def get_company_info(ticker: str) -> StockInfo:
    try:
        logger.info(f"Fetching company info for {ticker}")
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        if not info:
            return StockInfo(ticker=ticker.upper(), error="No company information available")
        
        return StockInfo(
            ticker=ticker.upper(),
            company_name=info.get('longName'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            market_cap=info.get('marketCap'),
            enterprise_value=info.get('enterpriseValue'),
            pe_ratio=info.get('trailingPE'),
            forward_pe=info.get('forwardPE'),
            peg_ratio=info.get('pegRatio'),
            price_to_book=info.get('priceToBook'),
            price_to_sales=info.get('priceToSalesTrailing12Months'),
            beta=info.get('beta'),
            dividend_rate=info.get('dividendRate'),
            dividend_yield=info.get('dividendYield'),
            payout_ratio=info.get('payoutRatio'),
            fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
            fifty_two_week_low=info.get('fiftyTwoWeekLow'),
            shares_outstanding=info.get('sharesOutstanding'),
            float_shares=info.get('floatShares'),
            employees=info.get('fullTimeEmployees'),
            currency=info.get('currency'),
            country=info.get('country'),
            website=info.get('website'),
            business_summary=info.get('longBusinessSummary')
        )
    
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {str(e)}")
        return StockInfo(ticker=ticker.upper(), error=str(e))

# New helper function to get dividends
def get_dividends_info(ticker: str) -> StockDividends:
    try:
        logger.info(f"Fetching dividends for {ticker}")
        stock = yf.Ticker(ticker.upper())
        dividends = stock.dividends
        
        if dividends.empty:
            return StockDividends(
                ticker=ticker.upper(), 
                dividends=[], 
                total_dividends=0.0,
                dividend_count=0,
                annual_dividend_rate=0.0
            )
        
        dividend_data = []
        for date, dividend in dividends.items():
            dividend_data.append(DividendData(
                date=date.strftime('%Y-%m-%d'),
                dividend=round(dividend, 4)
            ))
        
        total_dividends = round(dividends.sum(), 4)
        dividend_count = len(dividends)
        
        # Calculate annual dividend rate (last 4 quarters)
        annual_dividend_rate = 0.0
        if len(dividends) >= 4:
            annual_dividend_rate = round(dividends.tail(4).sum(), 4)
        elif len(dividends) > 0:
            # If less than 4 quarters, estimate based on available data
            annual_dividend_rate = round(dividends.tail(len(dividends)).sum(), 4)
        
        return StockDividends(
            ticker=ticker.upper(),
            dividends=dividend_data,
            total_dividends=total_dividends,
            dividend_count=dividend_count,
            annual_dividend_rate=annual_dividend_rate
        )
    
    except Exception as e:
        logger.error(f"Error fetching dividends for {ticker}: {str(e)}")
        return StockDividends(ticker=ticker.upper(), dividends=[], error=str(e))

# New helper function to get stock splits
def get_splits_info(ticker: str) -> StockSplits:
    try:
        logger.info(f"Fetching splits for {ticker}")
        stock = yf.Ticker(ticker.upper())
        splits = stock.splits
        
        if splits.empty:
            return StockSplits(
                ticker=ticker.upper(), 
                splits=[], 
                total_splits=0,
                cumulative_split_factor=1.0
            )
        
        split_data = []
        for date, split_ratio in splits.items():
            split_data.append(SplitData(
                date=date.strftime('%Y-%m-%d'),
                split_ratio=round(split_ratio, 4)
            ))
        
        total_splits = len(splits)
        cumulative_split_factor = round(splits.prod(), 4)
        
        return StockSplits(
            ticker=ticker.upper(),
            splits=split_data,
            total_splits=total_splits,
            cumulative_split_factor=cumulative_split_factor
        )
    
    except Exception as e:
        logger.error(f"Error fetching splits for {ticker}: {str(e)}")
        return StockSplits(ticker=ticker.upper(), splits=[], error=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AlphaFolio YFinance API",
        "version": "2.2.0",
        "endpoints": {
            "/current": "Get current stock data for multiple tickers (POST)",
            "/historical": "Get historical stock data for multiple tickers (POST)",
            "/info": "Get company information for multiple tickers (POST)",
            "/dividends": "Get dividend history for multiple tickers (POST)",
            "/splits": "Get stock split history for multiple tickers (POST)",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

# Current data endpoint - Multiple tickers (POST)
@app.post("/current", response_model=CurrentDataResponse)
async def get_current_data_post(request: TickersRequest):
    """Get current stock data for multiple tickers using POST method"""
    logger.info(f"Fetching current data for {len(request.tickers)} ticker(s)")
    
    stocks = []
    for ticker in request.tickers:
        stock_data = get_stock_info(ticker)
        stocks.append(stock_data)
    
    return CurrentDataResponse(stocks=stocks)

# Historical data endpoint - Multiple tickers (POST)
@app.post("/historical", response_model=HistoricalDataResponse)
async def get_historical_data_post(
    request: TickersRequest,
    period: str = Query("1mo", description="Period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="Interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
):
    """Get historical stock data for multiple tickers using POST method"""
    logger.info(f"Fetching historical data for {len(request.tickers)} ticker(s)")
    
    stocks = []
    for ticker in request.tickers:
        historical_data = get_historical_info(ticker, period, interval)
        stocks.append(historical_data)
    
    return HistoricalDataResponse(stocks=stocks, period=period)

# Stock info endpoint - Multiple tickers (POST)
@app.post("/info", response_model=StockInfoResponse)
async def get_stock_info_post(request: TickersRequest):
    """Get company information for multiple tickers using POST method"""
    logger.info(f"Fetching company info for {len(request.tickers)} ticker(s)")
    
    stocks = []
    for ticker in request.tickers:
        company_info = get_company_info(ticker)
        stocks.append(company_info)
    
    return StockInfoResponse(stocks=stocks)

# Dividends endpoint - Multiple tickers (POST)
@app.post("/dividends", response_model=DividendsResponse)
async def get_dividends_post(request: TickersRequest):
    """Get dividend history for multiple tickers using POST method"""
    logger.info(f"Fetching dividends for {len(request.tickers)} ticker(s)")
    
    stocks = []
    for ticker in request.tickers:
        dividends_data = get_dividends_info(ticker)
        stocks.append(dividends_data)
    
    return DividendsResponse(stocks=stocks)

# Stock splits endpoint - Multiple tickers (POST)
@app.post("/splits", response_model=SplitsResponse)
async def get_splits_post(request: TickersRequest):
    """Get stock split history for multiple tickers using POST method"""
    logger.info(f"Fetching splits for {len(request.tickers)} ticker(s)")
    
    stocks = []
    for ticker in request.tickers:
        splits_data = get_splits_info(ticker)
        stocks.append(splits_data)
    
    return SplitsResponse(stocks=stocks)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 