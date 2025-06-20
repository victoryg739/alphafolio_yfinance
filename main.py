from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
import time
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlphaFolio YFinance API",
    description="A unified backend server for fetching current and historical stock market data using Yahoo Finance",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache with TTL
class MemoryCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        
    def _generate_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_string = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str, ttl_seconds: int = 300) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            age = time.time() - self._timestamps[key]
            if age < ttl_seconds:
                logger.info(f"Cache HIT for key: {key[:20]}...")
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
        logger.info(f"Cache SET for key: {key[:20]}...")
    
    def clear_expired(self, ttl_seconds: int = 300) -> None:
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._timestamps[key]

# Global cache instance
cache = MemoryCache()

# Cache TTL settings (in seconds)
CACHE_TTL = {
    "current": 15 * 60,        # 15 minutes for current prices
    "historical": 24 * 60 * 60, # 24 hours for historical data
    "info": 7 * 24 * 60 * 60,   # 7 days for company info
    "dividends": 24 * 60 * 60,  # 24 hours for dividends
    "splits": 24 * 60 * 60,     # 24 hours for splits
}

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=10)

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
    timestamp: datetime = Field(default_factory=datetime.now)

class TickersRequest(BaseModel):
    tickers: List[str]

# New models for date-based historical requests
class TickerDateRequest(BaseModel):
    ticker: str
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

class HistoricalTickersRequest(BaseModel):
    tickers: List[TickerDateRequest]

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


# Enhanced helper functions with caching and async support

def get_stock_info_cached(ticker: str) -> StockPrice:
    """Get stock info with caching"""
    cache_key = cache._generate_key("current", ticker.upper())
    cached_result = cache.get(cache_key, CACHE_TTL["current"])
    
    if cached_result:
        return StockPrice(**cached_result)
    
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="2d")
        
        if hist.empty:
            result = StockPrice(ticker=ticker, error="No data available")
        else:
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
            
            change = None
            change_percent = None
            if current_price and previous_close:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
            
            result = StockPrice(
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
        
        # Cache the result
        cache.set(cache_key, result.dict())
        return result
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return StockPrice(ticker=ticker, error=str(e))

# Legacy function for backward compatibility
def get_stock_info(ticker: str) -> StockPrice:
    return get_stock_info_cached(ticker)

# Helper function to get historical data for a single ticker with date range
def get_historical_info(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> HistoricalStock:
    try:
        logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(start=start_date, end=end_date, interval=interval)
        
        if hist.empty:
            return HistoricalStock(ticker=ticker.upper(), data=[], error=f"No historical data found for {ticker} between {start_date} and {end_date}")
        
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



# NEW: Batch endpoint models
class BatchDataRequest(BaseModel):
    tickers: List[str]
    include_current: bool = True
    include_info: bool = True
    include_dividends: bool = True
    include_splits: bool = True
    include_historical: bool = False
    historical_start_date: Optional[str] = None
    historical_end_date: Optional[str] = None
    historical_interval: str = "1d"

class BatchTickerData(BaseModel):
    ticker: str
    current: Optional[StockPrice] = None
    info: Optional[StockInfo] = None
    dividends: Optional[StockDividends] = None
    splits: Optional[StockSplits] = None
    historical: Optional[HistoricalStock] = None
    processing_time_ms: Optional[float] = None

class BatchDataResponse(BaseModel):
    tickers: List[BatchTickerData]
    total_processing_time_ms: float
    cached_responses: int
    fresh_responses: int
    timestamp: datetime = Field(default_factory=datetime.now)

# Enhanced batch processing function
async def process_ticker_batch(ticker: str, request: BatchDataRequest) -> BatchTickerData:
    """Process all requested data for a single ticker with caching"""
    start_time = time.time()
    
    def run_in_thread(func, *args):
        return executor.submit(func, *args)
    
    # Submit all tasks to thread pool
    futures = {}
    
    if request.include_current:
        futures['current'] = run_in_thread(get_stock_info_cached, ticker)
    
    if request.include_info:
        futures['info'] = run_in_thread(get_company_info_cached, ticker)
    
    if request.include_dividends:
        futures['dividends'] = run_in_thread(get_dividends_info_cached, ticker)
    
    if request.include_splits:
        futures['splits'] = run_in_thread(get_splits_info_cached, ticker)
    
    if request.include_historical and request.historical_start_date and request.historical_end_date:
        futures['historical'] = run_in_thread(
            get_historical_info_cached, 
            ticker, 
            request.historical_start_date, 
            request.historical_end_date, 
            request.historical_interval
        )
    
    # Collect results
    results = {}
    for key, future in futures.items():
        try:
            results[key] = future.result(timeout=30)  # 30 second timeout
        except Exception as e:
            logger.error(f"Error processing {key} for {ticker}: {str(e)}")
            results[key] = None
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchTickerData(
        ticker=ticker,
        current=results.get('current'),
        info=results.get('info'),
        dividends=results.get('dividends'),
        splits=results.get('splits'),
        historical=results.get('historical'),
        processing_time_ms=processing_time
    )

# Add cached versions of other helper functions
def get_company_info_cached(ticker: str) -> StockInfo:
    """Get company info with caching"""
    cache_key = cache._generate_key("info", ticker.upper())
    cached_result = cache.get(cache_key, CACHE_TTL["info"])
    
    if cached_result:
        return StockInfo(**cached_result)
    
    result = get_company_info(ticker)  # Use existing function
    cache.set(cache_key, result.dict())
    return result

def get_dividends_info_cached(ticker: str) -> StockDividends:
    """Get dividends info with caching"""
    cache_key = cache._generate_key("dividends", ticker.upper())
    cached_result = cache.get(cache_key, CACHE_TTL["dividends"])
    
    if cached_result:
        return StockDividends(**cached_result)
    
    result = get_dividends_info(ticker)  # Use existing function
    cache.set(cache_key, result.dict())
    return result

def get_splits_info_cached(ticker: str) -> StockSplits:
    """Get splits info with caching"""
    cache_key = cache._generate_key("splits", ticker.upper())
    cached_result = cache.get(cache_key, CACHE_TTL["splits"])
    
    if cached_result:
        return StockSplits(**cached_result)
    
    result = get_splits_info(ticker)  # Use existing function
    cache.set(cache_key, result.dict())
    return result

def get_historical_info_cached(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> HistoricalStock:
    """Get historical info with caching"""
    cache_key = cache._generate_key("historical", ticker.upper(), start_date, end_date, interval)
    cached_result = cache.get(cache_key, CACHE_TTL["historical"])
    
    if cached_result:
        return HistoricalStock(**cached_result)
    
    result = get_historical_info(ticker, start_date, end_date, interval)  # Use existing function
    cache.set(cache_key, result.dict())
    return result

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AlphaFolio YFinance API",
        "version": "3.0.0",
        "endpoints": {
            "/current": "Get current stock data for multiple tickers (POST)",
            "/historical": "Get historical stock data for multiple tickers with date ranges (POST)",
            "/info": "Get company information for multiple tickers (POST)",
            "/dividends": "Get dividend history for multiple tickers (POST)",
            "/splits": "Get stock split history for multiple tickers (POST)",
            "/batch": "🚀 NEW: Get all data types for multiple tickers in one request (POST)",
            "/health": "Health check",
            "/cache/stats": "Cache statistics",
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

# Historical data endpoint - Multiple tickers with date ranges (POST)
@app.post("/historical", response_model=HistoricalDataResponse)
async def get_historical_data_post(
    request: HistoricalTickersRequest,
    interval: str = Query("1d", description="Interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
):
    """Get historical stock data for multiple tickers with custom date ranges using POST method"""
    logger.info(f"Fetching historical data for {len(request.tickers)} ticker(s) with date ranges")
    
    stocks = []
    for ticker_request in request.tickers:
        historical_data = get_historical_info(
            ticker_request.ticker, 
            ticker_request.start_date, 
            ticker_request.end_date, 
            interval
        )
        stocks.append(historical_data)
    
    return HistoricalDataResponse(stocks=stocks)

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



# NEW: High-performance batch endpoint
@app.post("/batch", response_model=BatchDataResponse)
async def get_batch_data(request: BatchDataRequest, background_tasks: BackgroundTasks):
    """🚀 Get all requested data for multiple tickers in parallel with caching"""
    logger.info(f"Batch request for {len(request.tickers)} tickers")
    start_time = time.time()
    
    # Process all tickers in parallel using asyncio
    tasks = [process_ticker_batch(ticker, request) for ticker in request.tickers]
    ticker_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    successful_results = []
    for i, result in enumerate(ticker_results):
        if isinstance(result, Exception):
            logger.error(f"Error processing ticker {request.tickers[i]}: {str(result)}")
            # Create error result
            error_result = BatchTickerData(
                ticker=request.tickers[i],
                processing_time_ms=0
            )
            successful_results.append(error_result)
        else:
            successful_results.append(result)
    
    total_time = (time.time() - start_time) * 1000
    
    # Count cache hits vs fresh fetches (simplified)
    cached_count = sum(1 for result in successful_results if result.processing_time_ms < 50)  # Assume <50ms = cache hit
    fresh_count = len(successful_results) - cached_count
    
    # Schedule cache cleanup in background
    background_tasks.add_task(cache.clear_expired)
    
    return BatchDataResponse(
        tickers=successful_results,
        total_processing_time_ms=total_time,
        cached_responses=cached_count,
        fresh_responses=fresh_count
    )

# Cache management endpoints
@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    total_entries = len(cache._cache)
    cache_size_mb = sum(len(str(v)) for v in cache._cache.values()) / (1024 * 1024)
    
    # Count entries by type
    entry_types = {}
    for key in cache._cache.keys():
        data_type = key.split('_')[0] if '_' in key else 'unknown'
        entry_types[data_type] = entry_types.get(data_type, 0) + 1
    
    return {
        "total_entries": total_entries,
        "cache_size_mb": round(cache_size_mb, 2),
        "entry_types": entry_types,
        "cache_ttl_settings": CACHE_TTL,
        "timestamp": datetime.now()
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    cache._cache.clear()
    cache._timestamps.clear()
    return {"message": "Cache cleared successfully", "timestamp": datetime.now()}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 