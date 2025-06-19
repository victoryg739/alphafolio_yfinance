"""
Test script to demonstrate the AlphaFolio YFinance API functionality
Run this script after starting the server to test all endpoints
"""

import requests
import json
import time
from typing import List, Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🏠 Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
        return False

def test_current_single_ticker_get():
    """Test getting current data for a single ticker using GET"""
    print("\n📈 Testing Current Data - Single Ticker (GET)...")
    try:
        ticker = "AAPL"
        response = requests.get(f"{BASE_URL}/current", params={"ticker": ticker})
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == 1
        assert data["stocks"][0]["ticker"] == ticker.upper()
        print("✅ Single ticker current data test passed!")
        return True
    except Exception as e:
        print(f"❌ Single ticker current data test failed: {e}")
        return False

def test_current_multiple_tickers_post():
    """Test getting current data for multiple tickers using POST"""
    print("\n📈 Testing Current Data - Multiple Tickers (POST)...")
    try:
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        payload = {"tickers": tickers}
        response = requests.post(f"{BASE_URL}/current", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == len(tickers)
        print("✅ Multiple tickers current data (POST) test passed!")
        return True
    except Exception as e:
        print(f"❌ Multiple tickers current data (POST) test failed: {e}")
        return False

def test_historical_single_ticker_get():
    """Test getting historical data for a single ticker using GET"""
    print("\n📊 Testing Historical Data - Single Ticker (GET)...")
    try:
        ticker = "AAPL"
        params = {
            "ticker": ticker,
            "period": "5d",
            "interval": "1d"
        }
        response = requests.get(f"{BASE_URL}/historical", params=params)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == 1
        assert data["stocks"][0]["ticker"] == ticker.upper()
        assert "data" in data["stocks"][0]
        print("✅ Single ticker historical data test passed!")
        return True
    except Exception as e:
        print(f"❌ Single ticker historical data test failed: {e}")
        return False

def test_historical_multiple_tickers_post():
    """Test getting historical data for multiple tickers using POST"""
    print("\n📊 Testing Historical Data - Multiple Tickers (POST)...")
    try:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        payload = {"tickers": tickers}
        params = {
            "period": "1mo",
            "interval": "1d"
        }
        response = requests.post(f"{BASE_URL}/historical", json=payload, params=params)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == len(tickers)
        print("✅ Multiple tickers historical data (POST) test passed!")
        return True
    except Exception as e:
        print(f"❌ Multiple tickers historical data (POST) test failed: {e}")
        return False

def test_info_single_ticker_get():
    """Test getting company info for a single ticker using GET"""
    print("\n🏢 Testing Company Info - Single Ticker (GET)...")
    try:
        ticker = "AAPL"
        response = requests.get(f"{BASE_URL}/info", params={"ticker": ticker})
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == 1
        assert data["stocks"][0]["ticker"] == ticker.upper()
        print("✅ Single ticker company info test passed!")
        return True
    except Exception as e:
        print(f"❌ Single ticker company info test failed: {e}")
        return False

def test_info_multiple_tickers_post():
    """Test getting company info for multiple tickers using POST"""
    print("\n🏢 Testing Company Info - Multiple Tickers (POST)...")
    try:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        payload = {"tickers": tickers}
        response = requests.post(f"{BASE_URL}/info", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == len(tickers)
        print("✅ Multiple tickers company info (POST) test passed!")
        return True
    except Exception as e:
        print(f"❌ Multiple tickers company info (POST) test failed: {e}")
        return False

def test_dividends_single_ticker_get():
    """Test getting dividends for a single ticker using GET"""
    print("\n💰 Testing Dividends - Single Ticker (GET)...")
    try:
        ticker = "AAPL"
        response = requests.get(f"{BASE_URL}/dividends", params={"ticker": ticker})
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == 1
        assert data["stocks"][0]["ticker"] == ticker.upper()
        assert "dividends" in data["stocks"][0]
        print("✅ Single ticker dividends test passed!")
        return True
    except Exception as e:
        print(f"❌ Single ticker dividends test failed: {e}")
        return False

def test_dividends_multiple_tickers_post():
    """Test getting dividends for multiple tickers using POST"""
    print("\n💰 Testing Dividends - Multiple Tickers (POST)...")
    try:
        tickers = ["AAPL", "MSFT", "KO"]  # Coca-Cola typically has good dividend history
        payload = {"tickers": tickers}
        response = requests.post(f"{BASE_URL}/dividends", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == len(tickers)
        print("✅ Multiple tickers dividends (POST) test passed!")
        return True
    except Exception as e:
        print(f"❌ Multiple tickers dividends (POST) test failed: {e}")
        return False

def test_splits_single_ticker_get():
    """Test getting stock splits for a single ticker using GET"""
    print("\n🔄 Testing Stock Splits - Single Ticker (GET)...")
    try:
        ticker = "AAPL"  # Apple has had stock splits
        response = requests.get(f"{BASE_URL}/splits", params={"ticker": ticker})
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == 1
        assert data["stocks"][0]["ticker"] == ticker.upper()
        assert "splits" in data["stocks"][0]
        print("✅ Single ticker splits test passed!")
        return True
    except Exception as e:
        print(f"❌ Single ticker splits test failed: {e}")
        return False

def test_splits_multiple_tickers_post():
    """Test getting stock splits for multiple tickers using POST"""
    print("\n🔄 Testing Stock Splits - Multiple Tickers (POST)...")
    try:
        tickers = ["AAPL", "TSLA", "AMZN"]  # Companies that have had splits
        payload = {"tickers": tickers}
        response = requests.post(f"{BASE_URL}/splits", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        assert "stocks" in data
        assert len(data["stocks"]) == len(tickers)
        print("✅ Multiple tickers splits (POST) test passed!")
        return True
    except Exception as e:
        print(f"❌ Multiple tickers splits (POST) test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🚨 Testing Error Handling...")
    
    # Test current endpoint with no ticker parameter
    try:
        response = requests.get(f"{BASE_URL}/current")
        print(f"No ticker param - Status Code: {response.status_code}")
        assert response.status_code == 422  # Validation error for missing required parameter
        print("✅ Missing ticker parameter error handling passed!")
    except Exception as e:
        print(f"❌ Missing ticker parameter error test failed: {e}")
        return False
    
    # Test with invalid ticker
    try:
        response = requests.get(f"{BASE_URL}/current", params={"ticker": "INVALID123"})
        print(f"Invalid ticker - Status Code: {response.status_code}")
        data = response.json()
        # Should return 200 but with error in the stock data
        assert response.status_code == 200
        assert data["stocks"][0]["error"] is not None
        print("✅ Invalid ticker error handling passed!")
    except Exception as e:
        print(f"❌ Invalid ticker error test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print("=" * 60)
    
    tests = [
        test_health_check,
        test_root_endpoint,
        test_current_single_ticker_get,
        test_current_multiple_tickers_post,
        test_historical_single_ticker_get,
        test_historical_multiple_tickers_post,
        test_info_single_ticker_get,
        test_info_multiple_tickers_post,
        test_dividends_single_ticker_get,
        test_dividends_multiple_tickers_post,
        test_splits_single_ticker_get,
        test_splits_multiple_tickers_post,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    print("📊 AlphaFolio YFinance API Test Suite")
    print("=" * 60)
    print("Make sure the API server is running on http://localhost:8000")
    print("Run: python main.py")
    print("=" * 60)
    
    # Wait a moment for user to start server if needed
    input("\nPress Enter to start tests (make sure server is running)...")
    
    success = run_all_tests()
    exit(0 if success else 1) 