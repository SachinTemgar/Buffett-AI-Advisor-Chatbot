import yfinance as yf
import pandas as pd
import pickle
import os

# Path to cached data
CACHE_PATH = os.path.join('data', 'demo_cache', 'stock_data.pkl')

def load_cached_data():
    """Load pre-downloaded stock data"""
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
    return {}

# Load cached data once at startup
_CACHED_STOCKS = load_cached_data()

def get_stock_financials(ticker):
    """
    Fetch all three financial statements for a given ticker.
    Returns: dict with income_statement, balance_sheet, cash_flow
    """
    # Try cached data first (silently)
    if ticker in _CACHED_STOCKS:
        cached = _CACHED_STOCKS[ticker]
        
        # Convert back to DataFrames
        return {
            'ticker': ticker,
            'info': cached['info'],
            'income_statement': pd.DataFrame(cached['financials']),
            'balance_sheet': pd.DataFrame(cached['balancesheet']),
            'cash_flow': pd.DataFrame(cached['cashflow']),
            'success': True
        }
    
    # If not in cache, try live yfinance (will likely fail on cloud, but works locally)
    try:
        stock = yf.Ticker(ticker)
        
        income = stock.financials
        balance = stock.balancesheet
        cashflow = stock.cashflow
        
        if income.empty:
            return {'success': False, 'error': "No financial data found"}

        return {
            'ticker': ticker,
            'info': stock.info,
            'income_statement': income,
            'balance_sheet': balance,
            'cash_flow': cashflow,
            'success': True
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }

def get_company_info(ticker):
    """Get basic company information including LIVE PRICE"""
    # Try cache first (silently)
    if ticker in _CACHED_STOCKS:
        info = _CACHED_STOCKS[ticker]['info']
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No summary available.'),
            'current_price': current_price
        }
    
    # Fallback to live yfinance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No summary available.'),
            'current_price': current_price
        }
    except:
        return {
            'name': ticker, 
            'sector': 'N/A', 
            'industry': 'N/A', 
            'market_cap': 'N/A',
            'summary': 'N/A',
            'current_price': 'N/A'
        }
